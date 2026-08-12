"""The ledger: every realized global reduction a STEAM run takes.

The cascade is local except for a handful of per-class, per-level (or
per-volume) SCALARS realized from the data: the joint product norm, the flux
means, the bounded add's solved per-level constants, and the composition's
projection. Those scalars are the only reason a size class cannot be evaluated
one tile at a time, so component 2 hoists them out of the per-class body into
an explicit MEASURE -> reduce -> APPLY seam and records them here.

Two things the format is built for.

**Additivity.** Every realized mean is stored as (total, count), never as the
mean. A mean is not additive across tiles; a sum is. Component 3's sweep 1
accumulates partials tile by tile, reduces, and hands the result to sweep 2.

**Dtype fidelity.** Two separate things, and conflating them costs the last
bits of the answer (caught by the bit-exactness gate, 2026-08-12):

- The ACCUMULATOR's dtype is part of the contract. The flux's mean-abs noise
  is summed in float32 and the volume means in float64, and those sums differ
  in value, so a total must be stored as the reduction produced it. See the
  caveat on `FluxAdvanceLedger.noise_abs`.
- The DIVISION is float64 in every case. That is not a choice: inline, a
  float32 total was divided by a numpy int64 count, and NEP 50 promotes that
  pair to float64 because an int64 is a strong type. Coercing the count to a
  Python int instead makes the pair float32 (a Python int is weak) and moves
  the result. `MeanReduction.mean` therefore divides in float64 explicitly,
  which reproduces the inline promotion without depending on it.

**Replay.** With a ledger in hand a class can be re-evaluated with every solve
skipped and every reduction injected, which reproduces the run bit-for-bit on
the same device. That is the primitive a streamed tile stands on, and it is
also what proves the seam is real rather than cosmetic.
"""

import numpy as np

# The bounded add's rescale loop is bounded by this many iterations
# (_bounded_amplitude_add's n_rescale default). The ledger's per-level arrays
# are sized to it; a caller passing a different n_rescale gets arrays that
# match its own value.
N_RESCALE_DEFAULT = 10


class MeanReduction:
    """An additive (total, count) pair standing in for a realized mean.

    ``total`` keeps the dtype the reduction produced -- float64 where the
    cascade accumulated in float64, float32 where it did not, since those sums
    differ in value. ``mean`` then divides in float64, which is what the inline
    code did by promotion (float32 over int64 -> float64) and what a float64
    total did directly.

    ``+`` is how a streamed sweep accumulates one tile's partial into the
    running total. It is exact for float64 totals; for a float32 total it is
    exact only to fp32 rounding, which is the documented tolerance of the
    streamed path (spec: "difference at the float32/FFT rounding floor").
    """

    __slots__ = ('total', 'count')

    def __init__(self, total, count):
        self.total = total
        self.count = int(count)

    @property
    def mean(self):
        # float() widens a float32 total exactly, so this is the float64
        # division the inline code performed -- see the module docstring on why
        # the division dtype must not be left to promotion.
        return float(self.total) / max(self.count, 1)

    def __add__(self, other):
        return MeanReduction(self.total + other.total,
                             self.count + other.count)

    def __repr__(self):
        return (f"MeanReduction(total={self.total!r} "
                f"[{np.asarray(self.total).dtype}], count={self.count})")


class PatternLedger:
    """The joint product norm of one scalar at one class.

    ``level_total`` is the float64 sum of |g W S| over the domain window at
    each level and ``level_count`` the number of turbulon centers there
    (nonzero cells). Both are per level and both additive across tiles.

    ``level_mean`` reproduces cascade_loop's expression exactly: divide by
    max(count, 1) so structureless levels give zero, then cast to float32,
    which is the dtype the normalization divides W by.
    """

    __slots__ = ('level_total', 'level_count')

    def __init__(self, level_total, level_count):
        self.level_total = np.asarray(level_total, dtype=np.float64)
        self.level_count = np.asarray(level_count, dtype=np.int64)

    @property
    def level_mean(self):
        return (self.level_total
                / np.maximum(self.level_count, 1)).astype(np.float32)

    def __add__(self, other):
        return PatternLedger(self.level_total + other.level_total,
                             self.level_count + other.level_count)


class FluxAdvanceLedger:
    """The three realized means of one flux advance, plus its clip count.

    ``entering`` and ``realized`` are volume means over the domain window,
    float64 totals, taken before the update and after the clip respectively;
    their ratio is the single scalar that corrects the clip's downward bias.

    ``noise_abs`` is the mean absolute multiplier noise over the turbulon
    centers, and its total is FLOAT32 -- `np.abs(noise_inner).sum()` with no
    dtype argument, which is what the cascade has always done. Recorded as it
    is, because component 2's gate is bit-exactness; but noted here because it
    is the one accumulator in this file that is not float64 while reducing over
    a field that reaches ~1e9 cells at the production finest class, and
    summing float32 partials across tiles is therefore exact only to fp32
    rounding. Reported to the coordinator rather than changed here.
    """

    __slots__ = ('entering', 'noise_abs', 'realized', 'n_clipped')

    def __init__(self, entering, noise_abs, realized, n_clipped):
        self.entering = entering
        self.noise_abs = noise_abs
        self.realized = realized
        self.n_clipped = int(n_clipped)

    @property
    def rescale(self):
        """The post-clip volume-mean restore factor, float32 as applied."""
        realized = self.realized.mean
        if realized > 0:
            return np.float32(self.entering.mean / realized)
        return None         # the whole domain clipped: caller flattens instead


class BoundedAddLedger:
    """The per-level scalars one bounded-add solve realized.

    Enough to replay the solve with NO reductions at all, which is the point:
    a streamed run solves on an assembled z-plane and then applies tile by
    tile, and every operation below is either a per-level scalar or a pointwise
    clip, so it tiles exactly.

    NOT reducible to a single (s, mu) pair, which is what the design spec
    assumed. The solve clips INSIDE its demean/rescale loop, so the composite
    map is a chain of affine-then-clip steps rather than clip(s*d - mu, caps);
    measured on a qt-like level against the lower bound, the single-pair form
    misses 58% of cells by up to 11% of the field scale. The SEQUENCE is what
    replays bit-exactly, so the sequence is what is recorded.

    Fields, all per level:
      a0            target amplitude <|d|>; <= 0 marks a clip-only level
      demean        (nz, n_rescale) loop demean shifts, in d-space
      scale         (nz, n_rescale) loop rescale factors
      n_loop        how many loop demeans ran (a per-level early exit)
      n_scale       how many rescales were applied (n_loop or n_loop - 1)
      final_demean  the demean after the loop
      mu            the bisected zero-mean shift; NaN where none was needed
    """

    __slots__ = ('a0', 'demean', 'scale', 'n_loop', 'n_scale',
                 'final_demean', 'mu')

    def __init__(self, nz, n_rescale=N_RESCALE_DEFAULT):
        self.a0 = np.zeros(nz, dtype=np.float64)
        self.demean = np.zeros((nz, n_rescale), dtype=np.float64)
        self.scale = np.ones((nz, n_rescale), dtype=np.float64)
        self.n_loop = np.zeros(nz, dtype=np.int32)
        self.n_scale = np.zeros(nz, dtype=np.int32)
        self.final_demean = np.zeros(nz, dtype=np.float64)
        self.mu = np.full(nz, np.nan, dtype=np.float64)

    @property
    def clip_only(self):
        """Levels with no amplitude to preserve: clipped, never rescaled."""
        return self.a0 <= 0.0

    @property
    def bisected(self):
        return ~np.isnan(self.mu)


class ProjectionLedger:
    """The final per-level projection of one scalar (_project_onto_bounds).

    ``mu`` is the mean-preserving shift per level, NaN where the level had no
    bound violation and was left bit-identical. The projection's target level
    mean lives only inside the bisection, so mu alone replays it.
    """

    __slots__ = ('mu',)

    def __init__(self, mu):
        self.mu = np.asarray(mu, dtype=np.float64)

    @property
    def violating(self):
        return ~np.isnan(self.mu)


class FluxOutputLedger:
    """_compose_flux_output's clip-and-restore: the same shape as one flux
    advance's, minus the noise (the composition draws none)."""

    __slots__ = ('entering', 'realized')

    def __init__(self, entering, realized):
        self.entering = entering
        self.realized = realized

    @property
    def rescale(self):
        realized = self.realized.mean
        if realized > 0:
            return np.float32(self.entering.mean / realized)
        return None


class ClassLedger:
    """One size class: the flux advance plus, per scalar, the product norm and
    the bounded add."""

    __slots__ = ('flux', 'pattern', 'bounded_add')

    def __init__(self):
        self.flux = None
        self.pattern = {}
        self.bounded_add = {}


class RunLedger:
    """A whole run: one ClassLedger per size class, plus the composition.

    ``classes`` is indexed by the run's OWN class index, so a nest's entry 0 is
    its first new class, not the root's. Composition entries are keyed by
    scalar name ('h', 'qt') and are filled by simulate()/refine() after the
    cascade returns, since the composition happens there.

    A RunLedger is both an output (measure mode records into it) and an input
    (apply mode reads from it). ``mode`` is not stored: which one it is is the
    caller's business, and a ledger recorded by one run is a valid input to
    the next.
    """

    __slots__ = ('classes', 'projection', 'flux_output')

    def __init__(self, n_classes):
        self.classes = [ClassLedger() for _ in range(n_classes)]
        self.projection = {}
        self.flux_output = None


# ---------------------------------------------------------------------------
# NetCDF serialization
# ---------------------------------------------------------------------------
#
# RAGGED VIA SUBGROUPS, following class_increments: nz differs from class to
# class (the grid shrinks as the padded extent does), and a padded (n_classes,
# nz_max) array plus a validity mask would put the reader in the business of
# knowing which entries are real. A subgroup per class carries its own z
# dimension and reads back without a mask.

LEDGER_GROUP = 'ledger'
_SCALARS = ('h', 'qt')


def write_ledger(dataset, run_ledger, group=None):
    """Write a RunLedger into an open netCDF4 Dataset as the `ledger` group."""
    if run_ledger is None:
        return
    parent = dataset if group is None else dataset[group]
    root = parent.createGroup(LEDGER_GROUP)
    root.description = (
        "Realized global reductions per size class: the scalars that make a "
        "class non-local. Sums and counts rather than means, so a streamed "
        "run can accumulate them tile by tile; enough to replay the run with "
        "every solve skipped."
    )
    root.n_classes = np.int32(len(run_ledger.classes))

    for index, class_ledger in enumerate(run_ledger.classes):
        grp = root.createGroup(f"c{index:02d}")
        flux = class_ledger.flux
        if flux is not None:
            grp.flux_entering_total = np.float64(flux.entering.total)
            grp.flux_entering_count = np.int64(flux.entering.count)
            # float32 on purpose: see FluxAdvanceLedger.
            grp.flux_noise_abs_total = np.float32(flux.noise_abs.total)
            grp.flux_noise_abs_count = np.int64(flux.noise_abs.count)
            grp.flux_realized_total = np.float64(flux.realized.total)
            grp.flux_realized_count = np.int64(flux.realized.count)
            grp.flux_n_clipped = np.int64(flux.n_clipped)

        any_pattern = next(iter(class_ledger.pattern.values()), None)
        if any_pattern is None:
            continue
        nz = len(any_pattern.level_total)
        grp.createDimension("z", nz)
        n_rescale = class_ledger.bounded_add[_SCALARS[0]].demean.shape[1]
        grp.createDimension("rescale", n_rescale)

        for name in _SCALARS:
            pattern = class_ledger.pattern[name]
            grp.createVariable(f"{name}_pattern_total", "f8", ("z",))[:] = \
                pattern.level_total
            grp.createVariable(f"{name}_pattern_count", "i8", ("z",))[:] = \
                pattern.level_count

            solve = class_ledger.bounded_add[name]
            grp.createVariable(f"{name}_a0", "f8", ("z",))[:] = solve.a0
            grp.createVariable(f"{name}_demean", "f8",
                               ("z", "rescale"))[:] = solve.demean
            grp.createVariable(f"{name}_scale", "f8",
                               ("z", "rescale"))[:] = solve.scale
            grp.createVariable(f"{name}_n_loop", "i4", ("z",))[:] = solve.n_loop
            grp.createVariable(f"{name}_n_scale", "i4", ("z",))[:] = solve.n_scale
            grp.createVariable(f"{name}_final_demean", "f8", ("z",))[:] = \
                solve.final_demean
            # NaN is the "no bisection" marker, so the variable must not carry
            # a fill value that would swallow it on read-back.
            mu = grp.createVariable(f"{name}_mu", "f8", ("z",),
                                    fill_value=False)
            mu[:] = solve.mu

    composition = root.createGroup("composition")
    if run_ledger.flux_output is not None:
        composition.flux_entering_total = np.float64(
            run_ledger.flux_output.entering.total)
        composition.flux_entering_count = np.int64(
            run_ledger.flux_output.entering.count)
        composition.flux_realized_total = np.float64(
            run_ledger.flux_output.realized.total)
        composition.flux_realized_count = np.int64(
            run_ledger.flux_output.realized.count)
    for name, projection in run_ledger.projection.items():
        if "z" not in composition.dimensions:
            composition.createDimension("z", len(projection.mu))
        variable = composition.createVariable(f"{name}_projection_mu", "f8",
                                              ("z",), fill_value=False)
        variable[:] = projection.mu


def read_ledger(dataset, group=None):
    """Read back what write_ledger wrote. Returns None if there is no ledger."""
    parent = dataset if group is None else dataset[group]
    if LEDGER_GROUP not in parent.groups:
        return None
    root = parent.groups[LEDGER_GROUP]
    run_ledger = RunLedger(int(root.n_classes))

    for index in range(int(root.n_classes)):
        grp = root.groups[f"c{index:02d}"]
        class_ledger = run_ledger.classes[index]
        if hasattr(grp, 'flux_entering_total'):
            class_ledger.flux = FluxAdvanceLedger(
                MeanReduction(np.float64(grp.flux_entering_total),
                              int(grp.flux_entering_count)),
                MeanReduction(np.float32(grp.flux_noise_abs_total),
                              int(grp.flux_noise_abs_count)),
                MeanReduction(np.float64(grp.flux_realized_total),
                              int(grp.flux_realized_count)),
                int(grp.flux_n_clipped),
            )
        if "z" not in grp.dimensions:
            continue
        nz = len(grp.dimensions["z"])
        n_rescale = len(grp.dimensions["rescale"])
        for name in _SCALARS:
            class_ledger.pattern[name] = PatternLedger(
                grp.variables[f"{name}_pattern_total"][:],
                grp.variables[f"{name}_pattern_count"][:])
            solve = BoundedAddLedger(nz, n_rescale)
            solve.a0[:] = grp.variables[f"{name}_a0"][:]
            solve.demean[:] = grp.variables[f"{name}_demean"][:]
            solve.scale[:] = grp.variables[f"{name}_scale"][:]
            solve.n_loop[:] = grp.variables[f"{name}_n_loop"][:]
            solve.n_scale[:] = grp.variables[f"{name}_n_scale"][:]
            solve.final_demean[:] = grp.variables[f"{name}_final_demean"][:]
            solve.mu[:] = grp.variables[f"{name}_mu"][:]
            class_ledger.bounded_add[name] = solve

    if "composition" in root.groups:
        composition = root.groups["composition"]
        if hasattr(composition, 'flux_entering_total'):
            run_ledger.flux_output = FluxOutputLedger(
                MeanReduction(np.float64(composition.flux_entering_total),
                              int(composition.flux_entering_count)),
                MeanReduction(np.float64(composition.flux_realized_total),
                              int(composition.flux_realized_count)),
            )
        for name in _SCALARS:
            key = f"{name}_projection_mu"
            if key in composition.variables:
                run_ledger.projection[name] = ProjectionLedger(
                    np.asarray(composition.variables[key][:], dtype=np.float64))
    return run_ledger
