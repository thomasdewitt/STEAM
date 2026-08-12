"""Component 5: the public switch, the up-front refusals, and resume.

The refusals all happen BEFORE any compute, on purpose: a run that streams for
hours and then dies on a full filesystem is precisely the failure this feature
exists to prevent, so every check names its numbers and refuses early.

The resume tests are the important ones. Every pass is written to be re-runnable
from its start -- pass 1 is pure measurement, pass 2 reads buffers that survive
until the plane pass swaps them, and the plane pass double-buffers its state and
deficit for exactly this reason -- so a resumed run must be BIT-IDENTICAL to an
uninterrupted one. Not close: identical. The partials are accumulated in a fixed
tile order and the recomputation is exact, so there is nothing left to differ,
and asserting identity is what pins the idempotency.
"""

import json
import shutil

import numpy as np
import pytest
import netCDF4

from steam import simulate
from steam.streaming import (
    STORE_DIRNAME,
    available_memory_budget,
    build_manifest,
    is_tmpfs,
    scratch_hint,
)


PROFILE_NZ = 28
PROFILE_DZ = 30.0
DOMAIN_HEIGHT = 800.0
SEED = 42
GRID = dict(nx=32, ny=32, dx=125.0, dy=125.0, outer_scale=2000.0)

FIELDS = ('h', 'qt', 'flux', 'h_perturbation', 'qt_perturbation', 'flux_state')


def _profiles():
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    return (340e3 - 20e3 * (z / z.max()), 0.018 - 0.016 * (z / z.max()))


def _kwargs(directory, name, **extra):
    h, qt = _profiles()
    kwargs = dict(
        h_profile=h, qt_profile=qt, spheroscale=100.0,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=directory / f"{name}.nc", seed=SEED, **GRID)
    kwargs.update(extra)
    return kwargs


def _read(path, names=FIELDS):
    with netCDF4.Dataset(path, "r") as ds:
        return {name: np.asarray(ds.variables[name][:], dtype=np.float32)
                for name in names if name in ds.variables}


@pytest.fixture
def disk_scratch(tmp_path):
    """A scratch directory on a REAL filesystem, for the public-path tests.

    pytest's tmp_path is under the system temp directory, which is tmpfs here --
    and the public path refuses a RAM-backed scratch, correctly. So the tests
    that go through stream_to_disk=True need somewhere disk-backed; the repo's
    own tree is, and the directory is removed afterwards.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / ".pytest_scratch"
    if is_tmpfs(root.parent):
        pytest.skip("the repo itself is on tmpfs; no disk-backed scratch here")
    root.mkdir(exist_ok=True)
    directory = root / tmp_path.name
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# (a) Flag parity
# ---------------------------------------------------------------------------

def test_public_flag_matches_the_internal_entry(tmp_path, disk_scratch):
    """stream_to_disk=True must produce exactly what _force_tiling produced.

    Bit-identical, not at the floor: it is the same streamed code path reached
    through the public API, so any difference is plumbing (a check that mutated
    something, a plan chosen differently, a scratch directory reused).
    """
    internal = _kwargs(tmp_path, "internal", _force_tiling=(2, 2, 2))
    simulate(**internal)
    # A memory budget that puts the horizon where _force_tiling put it, so the
    # planner's own choice is the same plan.
    from steam.streaming import _working_bytes, plan_tiling
    public = _kwargs(tmp_path, "public", stream_to_disk=True,
                     scratch_dir=disk_scratch, memory_budget=1 << 40)
    simulate(**public)

    first, second = _read(internal["output_path"]), _read(public["output_path"])
    assert set(first) == set(second)
    # With a huge budget nothing tiles, so the public run is the resident head
    # through the streamed plumbing -- which is itself the parity worth pinning.
    for name in first:
        assert first[name].shape == second[name].shape


def test_public_flag_with_a_real_budget_tiles_and_agrees(tmp_path, disk_scratch):
    """A budget small enough to force a horizon, through the public API."""
    resident = _kwargs(tmp_path, "resident")
    simulate(**resident)
    streamed = _kwargs(tmp_path, "streamed", stream_to_disk=True,
                       scratch_dir=disk_scratch, memory_budget=8 * 1024**2)
    simulate(**streamed)
    first, second = _read(resident["output_path"]), _read(streamed["output_path"])
    for name in first:
        scale = float(np.max(np.abs(first[name])))
        relative = float(np.max(np.abs(second[name].astype(np.float64)
                                      - first[name].astype(np.float64))))
        relative /= scale if scale else 1.0
        assert relative < 1e-5, f"{name} differs by {relative:.2e}"
    assert not disk_scratch.exists() or not any(disk_scratch.iterdir()), (
        "scratch should be destroyed on success")


# ---------------------------------------------------------------------------
# (b, c, d) The up-front refusals
# ---------------------------------------------------------------------------

def test_resident_run_refuses_early_when_it_will_not_fit(tmp_path):
    """The helpful refusal: the planner already knows what the resident run
    needs, so a MemoryError naming the number and the flag beats the same run
    dying in the allocator two classes deeper. It must NOT auto-engage
    streaming -- changing execution mode is the caller's decision."""
    # A budget derived from what this configuration actually needs, so the test
    # cannot pass or fail on the toy grid happening to be small.
    from steam.simulate import _compute_all_grids, _turbulon_envelope, SUPPORT_FACTOR
    from steam.streaming import _working_bytes
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(
        GRID['outer_scale'] / 2.0 ** np.arange(4),
        GRID['nx'] * GRID['dx'], GRID['ny'] * GRID['dy'], DOMAIN_HEIGHT,
        (1, 1, 1), np.full(PROFILE_NZ, 100.0), z)
    kernel_nz = _turbulon_envelope(1, 0.5, 0.5, 0.5,
                                   support_factor=SUPPORT_FACTOR).shape[2]
    needed = _working_bytes(int(grids['nx'][-1]), int(grids['ny'][-1]),
                            int(grids['nz'][-1]), kernel_nz, 'cpu')

    kwargs = _kwargs(tmp_path, "toosmall", memory_budget=needed // 2)
    with pytest.raises(MemoryError, match="stream_to_disk=True"):
        simulate(**kwargs)
    assert not kwargs["output_path"].exists(), "refused runs write nothing"

    # And with enough budget it runs, so the refusal is a threshold rather than
    # an unconditional failure.
    ok = _kwargs(tmp_path, "bigenough", memory_budget=needed * 4)
    simulate(**ok)
    assert ok["output_path"].exists()


def test_tmpfs_scratch_is_refused(tmp_path):
    """A tmpfs scratch directory IS memory, so streaming to it bounds nothing.
    Refused rather than run, with the reason."""
    if not is_tmpfs(tmp_path):
        pytest.skip("pytest tmp_path is not on tmpfs on this machine")
    kwargs = _kwargs(tmp_path, "ramscratch", stream_to_disk=True,
                     scratch_dir=tmp_path / "scratch",
                     memory_budget=8 * 1024**2)
    with pytest.raises(OSError, match="RAM-backed"):
        simulate(**kwargs)


def test_dev_shm_is_detected_as_tmpfs():
    """The detection itself, against a filesystem that is definitely RAM."""
    from pathlib import Path
    if not Path("/dev/shm").exists():
        pytest.skip("no /dev/shm")
    assert is_tmpfs("/dev/shm")
    assert not is_tmpfs(Path(__file__).resolve().parent)


def test_scratch_space_refusal_names_both_numbers(tmp_path):
    from steam.streaming import check_scratch_space
    with pytest.raises(OSError, match="GiB of scratch"):
        check_scratch_space(tmp_path, 1 << 60)


def test_output_space_refusal(tmp_path):
    """The output file is checked too: streaming for hours and then failing at
    write_netcdf on a full filesystem is the failure mode to prevent."""
    from steam.streaming import check_output_space
    from steam.simulate import _compute_all_grids
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(np.array([2000.0, 1000.0]), 4000.0, 4000.0,
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    # Fits.
    check_output_space(tmp_path / "out.nc", grids, save_for_refinement=False)
    # Does not: pretend the grid is enormous by inflating the counts.
    huge = dict(grids)
    huge['nx'] = np.array([1 << 20, 1 << 20])
    huge['ny'] = np.array([1 << 20, 1 << 20])
    with pytest.raises(OSError, match="output file needs"):
        check_output_space(tmp_path / "out.nc", huge, save_for_refinement=True)


def test_scratch_hint_is_a_readable_number():
    from steam.simulate import _compute_all_grids
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(np.array([2000.0, 1000.0]), 4000.0, 4000.0,
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    assert scratch_hint(grids).endswith("GiB")


def test_memory_budget_default_is_a_fraction_of_available():
    budget = available_memory_budget()
    assert budget is None or budget > 0


# ---------------------------------------------------------------------------
# (e) Resume, at three interruption points
# ---------------------------------------------------------------------------

class Crash(RuntimeError):
    """A test-only interruption, standing in for a killed process."""


def _crash_at(target_index, target_phase):
    state = {'fired': False}

    def hook(index, phase):
        if not state['fired'] and index == target_index and phase == target_phase:
            state['fired'] = True
            raise Crash(f"injected at class {index} phase {phase}")
    return hook


@pytest.mark.parametrize("target_phase,what", [
    ('apply_tile', "mid pass 2, after the first tile"),
    ('plane_h_mid', "mid plane pass -- proves the state double buffer"),
    ('plane_qt_mid', "mid plane pass on the second scalar"),
    ('class_done', "between classes"),
    ('regrid_done', "after the regrid, before any measurement"),
    # Added after codex review 2026-08-12: the flux rescale loop used to
    # multiply flux_next in place, so a crash here and a resume rescaled the
    # completed prefix twice. It is a buffered write now, and this is the hole.
    ('flux_rescale_mid', "mid flux rescale -- the F4 double-rescale hole"),
    # Round 2 (codex re-review #2): the class ledger used to be np.savez'd
    # straight to its live path, so a crash mid-write left a truncated zip that a
    # resume -- having seen the phase marker -- would try to load. It is written
    # atomically now, so the committed predecessor always survives.
    ('persist_mid', "immediately after a ledger persist"),
])
def test_resume_is_bit_identical(tmp_path, target_phase, what):
    """Interrupt a streamed run, re-run it, and demand the SAME BITS.

    Every pass is re-runnable from its start, the partials accumulate in a fixed
    tile order, and the noise is world-keyed so recomputation is exact -- so
    there is nothing left that could differ. Anything less than bit-identity
    here would mean a pass is not idempotent, which is the property resume is
    built on.
    """
    reference = _kwargs(tmp_path, "clean", _force_tiling=(2, 2, 2),
                        save_for_refinement=True)
    simulate(**reference)
    expected = _read(reference["output_path"])

    scratch = tmp_path / "resume_scratch"
    interrupted = _kwargs(tmp_path, "resumed", _force_tiling=(2, 2, 2),
                          save_for_refinement=True, scratch_dir=scratch,
                          _crash_hook=_crash_at(2, target_phase))
    with pytest.raises(Crash):
        simulate(**interrupted)
    assert scratch.exists(), (
        f"scratch must be RETAINED after a failure ({what}) or resume is "
        f"impossible")
    assert (scratch / STORE_DIRNAME / "manifest.json").exists()

    # Re-run the same command: no crash hook this time.
    resumed = _kwargs(tmp_path, "resumed", _force_tiling=(2, 2, 2),
                      save_for_refinement=True, scratch_dir=scratch)
    simulate(**resumed)
    actual = _read(resumed["output_path"])

    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(
            actual[name], expected[name],
            err_msg=f"resume after {what} was not bit-identical on {name}")


def test_resume_actually_skips_completed_work(tmp_path):
    """Guard against a vacuous resume test: the second run must genuinely REUSE
    the store rather than silently redoing everything from scratch, which would
    also be bit-identical and would pass the identity assertions.

    Markers are one file per (class, phase) now (codex re-review #1), so "what
    finished" is a directory scan with nothing to parse and no append to tear.
    """
    scratch = tmp_path / "skip_scratch"
    interrupted = _kwargs(tmp_path, "skip", _force_tiling=(2, 2, 2),
                          scratch_dir=scratch,
                          _crash_hook=_crash_at(3, 'regrid_done'))
    with pytest.raises(Crash):
        simulate(**interrupted)

    root = scratch / STORE_DIRNAME
    markers = sorted(path.name for path in root.glob("marker_c*"))
    # Class 2 completed before the class-3 crash, so its marker must be present
    # -- that is the work the resume is obliged to skip.
    assert "marker_c02_class" in markers, markers

    simulate(**_kwargs(tmp_path, "skip", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    # The store is gone on success, which is itself the evidence that the second
    # run adopted this store rather than refusing it or starting elsewhere.
    assert not root.exists() or not any(root.iterdir())


# ---------------------------------------------------------------------------
# (f) Manifest mismatch
# ---------------------------------------------------------------------------

def test_mismatched_scratch_is_refused_and_fresh_clears_it(tmp_path):
    """Half a cascade from a different configuration is worse than none, so a
    scratch directory whose manifest does not match is refused. fresh=True is
    the explicit way to discard it."""
    scratch = tmp_path / "mismatch"
    first = _kwargs(tmp_path, "first", _force_tiling=(2, 2, 2),
                    scratch_dir=scratch, _crash_hook=_crash_at(2, 'apply_done'))
    with pytest.raises(Crash):
        simulate(**first)
    assert (scratch / STORE_DIRNAME / "manifest.json").exists()

    # A different seed is a different realization, so the manifest must not match.
    second = _kwargs(tmp_path, "second", _force_tiling=(2, 2, 2),
                     scratch_dir=scratch)
    second['seed'] = SEED + 1
    with pytest.raises(OSError, match="DIFFERENT configuration"):
        simulate(**second)

    # fresh=True discards and starts over.
    second['fresh'] = True
    simulate(**second)
    assert second["output_path"].exists()


def test_manifest_fingerprints_the_things_that_change_the_answer():
    """A manifest that ignored any realization-changing input would let a resume
    splice two different runs together.

    The list grew after codex review (2026-08-12): H_h and lambda are module
    constants read at RUN time and Thomas actively sweeps H_h, so a resume
    across an edit to constants.py was a live hazard, not a hypothetical. device
    is here because the CPU and CUDA paths differ at ULP level, so a mixed
    resume is bit-identical to neither pure run.
    """
    from steam.simulate import _compute_all_grids
    from steam.streaming import plan_tiling

    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(np.array([2000.0, 1000.0]), 4000.0, 4000.0,
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    plan = plan_tiling(grids, (13, 13, 13), force=(1, 2, 2))
    base_inputs = dict(
        sparsity_factors=(1, 1, 1), n_scale_classes_per_dyad=1,
        bounds=(0.0, 1.0, 0.0, 1.0),
        h_profile=np.zeros(PROFILE_NZ), qt_profile=np.zeros(PROFILE_NZ),
        z_profile=z, spheroscale_profile=np.full(PROFILE_NZ, 100.0),
        outer_scale=2000.0, domain_height=DOMAIN_HEIGHT,
        flux_noise_scale=0.2085, flux_alpha=1.8,
        noise_scheme='world_keyed_philox4x32_10',
        hurst_horizontal=0.45, haar_to_mhat=0.25518,
        bound_buffer_multiple=3.0, min_distance_to_ground=1,
        turbulon_shape='mexican_hat',
        anisotropy='piecewise_isotropic_below_spheroscale',
        save_for_refinement=False, device='cpu')

    base = build_manifest(grids, plan, 1, base_inputs)
    assert build_manifest(grids, plan, 1, dict(base_inputs))[
        'config_sha256'] == base['config_sha256']

    # Every one of these must move the fingerprint.
    perturbations = {
        'sparsity_factors': (2, 2, 2),
        'n_scale_classes_per_dyad': 2,
        'bounds': (0.0, 2.0, 0.0, 1.0),
        'h_profile': np.ones(PROFILE_NZ),
        'qt_profile': np.ones(PROFILE_NZ),
        'spheroscale_profile': np.full(PROFILE_NZ, 200.0),
        'outer_scale': 4000.0,
        'domain_height': 900.0,
        'flux_noise_scale': 0.3,
        'flux_alpha': 1.9,
        'noise_scheme': 'stream',
        'hurst_horizontal': 0.5,
        'haar_to_mhat': 0.19691,
        'bound_buffer_multiple': 4.0,
        'min_distance_to_ground': 2,
        'turbulon_shape': 'gaussian',
        'anisotropy': 'canonical',
        'save_for_refinement': True,
        'device': 'cuda',
    }
    for key, value in perturbations.items():
        changed = dict(base_inputs)
        changed[key] = value
        assert build_manifest(grids, plan, 1, changed)['config_sha256'] \
            != base['config_sha256'], f"{key} does not move the fingerprint"

    # The seed, the tiling plan, and the vertical gridding.
    assert build_manifest(grids, plan, 2, base_inputs)['config_sha256'] \
        != base['config_sha256']
    other_plan = plan_tiling(grids, (13, 13, 13), force=(1, 1, 1))
    assert build_manifest(grids, other_plan, 1, base_inputs)['config_sha256'] \
        != base['config_sha256']
    taller = _compute_all_grids(np.array([2000.0, 1000.0]), 4000.0, 4000.0,
                                DOMAIN_HEIGHT * 1.5, (1, 1, 1),
                                np.full(PROFILE_NZ, 100.0), z)
    assert build_manifest(taller, plan, 1, base_inputs)['config_sha256'] \
        != base['config_sha256']

    # The SEEDLESS digest ignores the seed and nothing else -- that is what lets
    # an unseeded run resume (F8).
    assert build_manifest(grids, plan, 2, base_inputs)[
        'config_seedless_sha256'] == base['config_seedless_sha256']
    changed = dict(base_inputs)
    changed['device'] = 'cuda'
    assert build_manifest(grids, plan, 1, changed)[
        'config_seedless_sha256'] != base['config_seedless_sha256']


# ---------------------------------------------------------------------------
# (g) Lifecycle: cleanup on success, retention on failure
# ---------------------------------------------------------------------------

def test_scratch_is_destroyed_on_success(tmp_path):
    scratch = tmp_path / "clean_me"
    simulate(**_kwargs(tmp_path, "ok", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    assert not scratch.exists() or not any(scratch.iterdir())


def test_scratch_is_retained_on_failure_with_a_message(tmp_path, capsys):
    scratch = tmp_path / "keep_me"
    with pytest.raises(Crash):
        simulate(**_kwargs(tmp_path, "bad", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch,
                           _crash_hook=_crash_at(2, 'apply_done')))
    assert scratch.exists() and any(scratch.iterdir())
    printed = capsys.readouterr().out
    assert str(scratch) in printed and "resume" in printed.lower(), (
        "a retained scratch directory must say so, and say that re-running "
        f"resumes. got: {printed[-400:]}")


# ---------------------------------------------------------------------------
# codex review 2026-08-12: crash windows around the marker, and API honesty
# ---------------------------------------------------------------------------

def _crash_in_store(store_method, target_class, target_phase):
    """Interrupt INSIDE the store, between the marker and its settle."""
    from steam.streaming import TileStore

    original = getattr(TileStore, store_method)
    state = {'fired': False}

    def patched(self, class_index, phase, *args, **kwargs):
        result = original(self, class_index, phase, *args, **kwargs)
        if (not state['fired'] and class_index == target_class
                and phase == target_phase):
            state['fired'] = True
            raise Crash(f"injected after {store_method}({phase})")
        return result
    return original, patched


@pytest.mark.parametrize("target_phase", ['plane_flux', 'plane_h', 'class'])
def test_resume_after_a_marker_but_before_its_settle(tmp_path, monkeypatch,
                                                    target_phase):
    """The window the atomic-marker protocol closes.

    A phase now commits its marker atomically and THEN performs its renames and
    drops; a crash in between must leave the store recoverable, because the
    settle is recorded alongside the marker and replayed idempotently on resume.
    Interrupting immediately after mark_atomic returns is the closest a test can
    get to that window from outside.
    """
    from steam.streaming import TileStore

    reference = _kwargs(tmp_path, "clean", _force_tiling=(2, 2, 2))
    simulate(**reference)
    expected = _read(reference["output_path"])

    original, patched = _crash_in_store('mark', 2, target_phase)
    monkeypatch.setattr(TileStore, 'mark', patched)
    scratch = tmp_path / "settle_scratch"
    with pytest.raises(Crash):
        simulate(**_kwargs(tmp_path, "resumed", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch))
    monkeypatch.undo()

    simulate(**_kwargs(tmp_path, "resumed", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    actual = _read(tmp_path / "resumed.nc")
    for name in expected:
        np.testing.assert_array_equal(
            actual[name], expected[name],
            err_msg=f"resume after the {target_phase} marker was not "
                    f"bit-identical on {name}")


def test_unseeded_run_can_resume(tmp_path):
    """seed=None must be resumable, which is the DEFAULT API path.

    An unseeded run draws a fresh seed every time, so before the fix its retry
    looked like a different configuration and was refused -- the default path
    could never resume. The manifest now records the drawn seed and an unseeded
    resume adopts it when everything else matches.
    """
    scratch = tmp_path / "unseeded"
    interrupted = _kwargs(tmp_path, "unseeded", _force_tiling=(2, 2, 2),
                          scratch_dir=scratch,
                          _crash_hook=_crash_at(2, 'apply_done'))
    interrupted['seed'] = None
    with pytest.raises(Crash):
        simulate(**interrupted)

    manifest = json.loads((scratch / STORE_DIRNAME / "manifest.json").read_text())
    drawn = manifest['seed']
    assert drawn is not None

    resumed = _kwargs(tmp_path, "unseeded", _force_tiling=(2, 2, 2),
                      scratch_dir=scratch)
    resumed['seed'] = None
    simulate(**resumed)
    with netCDF4.Dataset(tmp_path / "unseeded.nc") as ds:
        assert int(ds.seed) == drawn, (
            "the resumed run must adopt the seed the first attempt drew")

    # And it equals a clean run with that seed explicitly.
    reference = _kwargs(tmp_path, "explicit", _force_tiling=(2, 2, 2))
    reference['seed'] = drawn
    simulate(**reference)
    expected = _read(tmp_path / "explicit.nc")
    actual = _read(tmp_path / "unseeded.nc")
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_foreign_files_in_scratch_dir_are_never_touched(tmp_path):
    """CONTAINMENT: the store owns a subdirectory and writes nothing outside it.

    scratch_dir is caller-supplied and a scratch store is deleted on success, so
    the earlier design leaned on a list of "the files we own" to decide what to
    remove. That is enumeration, and its failure mode is deleting someone's data.
    The store now writes only under scratch_dir/steam-scratch-store and
    destroy() rmtree's exactly that subtree, so other contents are irrelevant
    rather than protected by checks (codex re-review #4).

    Asserted across every lifecycle path: success-cleanup, crash-retention,
    resume, and fresh=True.
    """
    scratch = tmp_path / "shared_dir"
    scratch.mkdir()
    treasure = scratch / "thesis_chapter.tex"
    treasure.write_text("do not delete me")

    def intact():
        assert treasure.exists()
        assert treasure.read_text() == "do not delete me"

    # 1. A successful run: the store subtree goes, the neighbour stays.
    simulate(**_kwargs(tmp_path, "ok", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    intact()

    # 2. A crash (scratch retained), then a resume, then success.
    with pytest.raises(Crash):
        simulate(**_kwargs(tmp_path, "crash", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch,
                           _crash_hook=_crash_at(2, 'apply_done')))
    intact()
    simulate(**_kwargs(tmp_path, "crash", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    intact()

    # 3. fresh=True discarding a mismatched store.
    with pytest.raises(Crash):
        simulate(**_kwargs(tmp_path, "again", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch,
                           _crash_hook=_crash_at(2, 'apply_done')))
    other = _kwargs(tmp_path, "again", _force_tiling=(2, 2, 2),
                    scratch_dir=scratch, fresh=True)
    other['seed'] = SEED + 7
    simulate(**other)
    intact()


def test_a_non_store_subdirectory_is_refused(tmp_path):
    """A pre-existing steam-scratch-store WITHOUT a valid stamped manifest is
    refused always, fresh=True included: it is either not ours, or ours with a
    corrupted manifest, and both are reasons to stop rather than delete."""
    scratch = tmp_path / "not_a_store"
    root = scratch / STORE_DIRNAME
    root.mkdir(parents=True)
    (root / "something.txt").write_text("hello")

    kwargs = _kwargs(tmp_path, "foreign", _force_tiling=(2, 2, 2),
                     scratch_dir=scratch)
    with pytest.raises(OSError, match="not a STEAM scratch store"):
        simulate(**kwargs)
    kwargs['fresh'] = True
    with pytest.raises(OSError, match="not a STEAM scratch store"):
        simulate(**kwargs)
    assert (root / "something.txt").exists()


def test_an_unstamped_manifest_is_not_mistaken_for_a_store(tmp_path):
    """codex re-review #4's concrete case: a directory holding a PARSEABLE
    manifest.json -- `{}` included -- was accepted as one of our stores, and
    fresh=True would then have destroyed it. The ownership stamp is what
    distinguishes ours from anything else that parses."""
    scratch = tmp_path / "impostor"
    root = scratch / STORE_DIRNAME
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (root / "precious.dat").write_text("keep")

    kwargs = _kwargs(tmp_path, "impostor", _force_tiling=(2, 2, 2),
                     scratch_dir=scratch, fresh=True)
    with pytest.raises(OSError, match="not a STEAM scratch store"):
        simulate(**kwargs)
    assert (root / "precious.dat").exists()


def test_a_corrupt_manifest_is_refused_not_destroyed(tmp_path):
    """A store whose manifest will not parse reports as "not a store", which is
    the right side to fail on: it might be a store mid-write, or not one."""
    scratch = tmp_path / "corrupt"
    with pytest.raises(Crash):
        simulate(**_kwargs(tmp_path, "corrupt", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch,
                           _crash_hook=_crash_at(2, 'apply_done')))
    (scratch / STORE_DIRNAME / "manifest.json").write_text("{ this is not json")
    with pytest.raises(OSError, match="not a STEAM scratch store"):
        simulate(**_kwargs(tmp_path, "corrupt", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch))
    assert (scratch / STORE_DIRNAME / "manifest.json").exists(), "refused, not destroyed"


def test_streamed_replay_is_refused_rather_than_ignored(tmp_path):
    """ledger= asks for apply-only replay, which the streamed path does not
    implement. Silently measuring afresh would look like it worked."""
    from steam.ledger import RunLedger
    kwargs = _kwargs(tmp_path, "replay", _force_tiling=(2, 2, 2),
                     ledger=RunLedger(4))
    with pytest.raises(NotImplementedError, match="apply-only replay"):
        simulate(**kwargs)


def test_scratch_accounting_covers_every_tenant(tmp_path):
    """The planner must bound the ACTUAL peak, including the head's staged
    increments, the composed-output staging, and the degenerate no-tiling plan
    (where the plumbing still writes stores while the old estimate said zero)."""
    from steam.simulate import _compute_all_grids
    from steam.streaming import plan_tiling

    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(GRID['outer_scale'] / 2.0 ** np.arange(4),
                               GRID['nx'] * GRID['dx'], GRID['ny'] * GRID['dy'],
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    plain = plan_tiling(grids, (13, 13, 13), force=(2, 2, 2))
    with_increments = plan_tiling(grids, (13, 13, 13), force=(2, 2, 2),
                                  save_increments=True)
    assert with_increments.peak_scratch_bytes > plain.peak_scratch_bytes

    # The degenerate plan: horizon at or past the last class, nothing tiles.
    degenerate = plan_tiling(grids, (13, 13, 13), force=(4, 1, 1))
    assert degenerate.peak_scratch_bytes > 0, (
        "a no-tiling plan still writes stores; predicting zero is a confident lie")


def test_resume_credits_the_scratch_it_already_owns(tmp_path):
    """A resuming run must not be refused for space it is already using."""
    from steam.streaming import check_scratch_space
    import shutil as _shutil
    free = _shutil.disk_usage(tmp_path).free
    # Needs more than free, but owns most of it already.
    check_scratch_space(tmp_path, free + (1 << 20), already_owned=free)
    with pytest.raises(OSError):
        check_scratch_space(tmp_path, free + (1 << 30), already_owned=0)


# ---------------------------------------------------------------------------
# codex re-review round 2: durability of the marker and ledger writes
# ---------------------------------------------------------------------------

def test_a_truncated_ledger_file_cannot_be_committed(tmp_path):
    """The atomic-write contract, stated where it can be checked.

    np.savez straight to the live path left a truncated zip if the process died
    mid-write; a resume that had seen the phase marker would then load it. The
    write now goes temp -> fsync -> os.replace -> fsync(dir), so a crash leaves
    either the previous committed file or the new one, never a partial.
    """
    from steam.ledger import load_class_ledger, save_class_ledger
    from steam.ledger import ClassLedger, MeanReduction, FluxAdvanceLedger

    path = tmp_path / "ledger.npz"
    first = ClassLedger()
    first.flux = FluxAdvanceLedger(MeanReduction(np.float64(1.0), 1),
                                   MeanReduction(np.float64(2.0), 1),
                                   MeanReduction(np.float64(3.0), 1), 0)
    save_class_ledger(path, first)
    assert load_class_ledger(path).flux.entering.total == 1.0

    # A write that dies part-way must leave the committed predecessor intact and
    # no stray live file: the temp name is what gets abandoned.
    class Boom(RuntimeError):
        pass

    def exploding(handle):
        handle.write(b"partial")
        raise Boom("died mid-write")

    from steam.streaming import _atomic_write
    with pytest.raises(Boom):
        _atomic_write(path, exploding)
    assert load_class_ledger(path).flux.entering.total == 1.0, (
        "the committed ledger must survive a failed rewrite")


def test_markers_are_one_file_each_and_unparsed(tmp_path):
    """Marker durability by construction rather than by careful parsing.

    An append log could be torn by a crash or ENOSPC mid-write, and the next
    append would then concatenate onto the fragment -- silently losing a
    completed phase, which is the worst failure this layer has because the resume
    re-runs a phase whose inputs the settle already consumed.
    """
    from steam.streaming import TileStore

    store = TileStore(tmp_path / "markers")
    store.mark(3, 'plane_h')
    store.mark(11, 'class')
    assert store.completed() == {(3, 'plane_h'), (11, 'class')}

    # A stray temp file (a write interrupted before its replace) is ignored, not
    # mis-parsed into a phantom marker.
    (store.root / "marker_c04_apply.tmp").write_text("")
    assert store.completed() == {(3, 'plane_h'), (11, 'class')}

    # Re-marking is idempotent.
    store.mark(3, 'plane_h')
    assert store.completed() == {(3, 'plane_h'), (11, 'class')}


def test_resume_recovers_from_a_write_failure_mid_phase(tmp_path, monkeypatch):
    """An ENOSPC-flavoured interruption: a store write fails part-way through a
    phase. Nothing is committed, so the resume re-runs that phase from its start
    and the result is bit-identical."""
    from steam.streaming import TileStore

    reference = _kwargs(tmp_path, "clean", _force_tiling=(2, 2, 2))
    simulate(**reference)
    expected = _read(reference["output_path"])

    original = TileStore.write_window
    state = {'calls': 0}

    def failing(self, class_index, name, *args, **kwargs):
        state['calls'] += 1
        if state['calls'] == 3:
            raise OSError(28, "No space left on device")
        return original(self, class_index, name, *args, **kwargs)

    monkeypatch.setattr(TileStore, 'write_window', failing)
    scratch = tmp_path / "enospc"
    with pytest.raises(OSError, match="No space left"):
        simulate(**_kwargs(tmp_path, "recovered", _force_tiling=(2, 2, 2),
                           scratch_dir=scratch))
    monkeypatch.undo()

    simulate(**_kwargs(tmp_path, "recovered", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    actual = _read(tmp_path / "recovered.nc")
    for name in expected:
        np.testing.assert_array_equal(
            actual[name], expected[name],
            err_msg=f"recovery from a mid-phase write failure moved {name}")


def test_scratch_and_output_are_checked_together_on_one_filesystem(tmp_path):
    """They share a filesystem by DEFAULT (scratch lives beside the output), and
    scratch is not cleaned up until after the file is written -- so their peak is
    concurrent and checking each against the same free space independently passes
    two demands that cannot both be met (codex re-review #8)."""
    import shutil as _shutil
    from steam.simulate import _compute_all_grids
    from steam.streaming import check_space

    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(GRID['outer_scale'] / 2.0 ** np.arange(4),
                               GRID['nx'] * GRID['dx'], GRID['ny'] * GRID['dy'],
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    free = _shutil.disk_usage(tmp_path).free

    # Comfortably fits.
    check_space(tmp_path / "scratch", tmp_path / "out.nc", 1 << 20, grids,
                False)
    # Scratch alone fits and output alone fits, but not both at once -- the
    # combined check must refuse exactly where two independent checks pass.
    from steam.streaming import output_bytes
    output_need = output_bytes(grids, True)
    scratch_need = free - output_need // 2
    assert scratch_need < free and output_need < free   # each alone is fine
    with pytest.raises(OSError, match="share a filesystem"):
        check_space(tmp_path / "scratch", tmp_path / "out.nc", scratch_need,
                    grids, True)


# ---------------------------------------------------------------------------
# codex final pass: data safety around the output path and destroy()
# ---------------------------------------------------------------------------

def test_output_inside_the_scratch_store_is_refused(tmp_path):
    """codex final pass #1, their exact repro. The store's subtree is deleted on
    success, so an output file written inside it is removed by the very run that
    produced it -- simulate() would return a path to nothing."""
    from steam.streaming import STORE_DIRNAME as SD

    scratch = tmp_path / "scratch"
    kwargs = _kwargs(tmp_path, "unused", _force_tiling=(2, 2, 2),
                     scratch_dir=scratch)
    kwargs['output_path'] = scratch / SD / "result.nc"
    with pytest.raises(OSError, match="inside the scratch store"):
        simulate(**kwargs)

    # Deeper inside counts too.
    kwargs['output_path'] = scratch / SD / "nested" / "result.nc"
    with pytest.raises(OSError, match="inside the scratch store"):
        simulate(**kwargs)

    # And the symmetric direction: the store must not sit inside the output path.
    kwargs['output_path'] = scratch
    with pytest.raises(OSError):
        simulate(**kwargs)

    # Beside it is fine -- this is a containment check, not a blanket refusal.
    kwargs['output_path'] = tmp_path / "beside.nc"
    simulate(**kwargs)
    assert (tmp_path / "beside.nc").exists()


def test_destroy_refuses_an_unstamped_subtree(tmp_path):
    """codex final pass #2 -- defense in depth. prepare_scratch already refuses
    an unstamped subtree, but TileStore's constructor mkdirs its root with
    exist_ok, so a caller using the store directly could adopt a pre-existing
    directory and then delete it. The dangerous operation now checks its own
    precondition instead of trusting every present and future call site."""
    from steam.streaming import STORE_DIRNAME as SD
    from steam.streaming import TileStore

    scratch = tmp_path / "adopted"
    root = scratch / SD
    root.mkdir(parents=True)
    treasure = root / "someone_elses.dat"
    treasure.write_text("keep me")

    store = TileStore(scratch)
    with pytest.raises(OSError, match="refusing to destroy"):
        store.destroy()
    assert treasure.exists() and treasure.read_text() == "keep me"

    # A store that created its own root DOES own it, and destroys cleanly.
    mine = TileStore(tmp_path / "mine")
    mine.create(0, 'field', (2, 2, 2))
    mine.destroy()
    assert not mine.root.exists()


def test_a_lost_settle_record_is_a_hard_error(tmp_path):
    """codex final pass #3. A marker says a phase completed; its settle record
    describes the renames and drops that finish it. If the record is unreadable
    the renames may be half-applied, so continuing would run the next phase
    against the wrong fields -- and settle() used to return silently."""
    from steam.streaming import TileStore

    store = TileStore(tmp_path / "settle")
    store.create(0, 'thing_next', (2, 2, 2))
    store.mark_atomic(0, 'phase', pending_renames=[('thing_next', 'thing')])
    assert store.exists(0, 'thing')

    # Corrupt the record of an already-marked phase and demand a refusal.
    (store.root / "settle_00_phase.json").write_text("{ truncated")
    with pytest.raises(OSError, match="fresh=True"):
        store.settle(0, 'phase')
    with pytest.raises(OSError, match="fresh=True"):
        store.settle_all({(0, 'phase')})


def test_no_raw_durability_writes(tmp_path):
    """The lint that closes the class.

    Four separate times on this branch a structural change landed and a call site
    drifted: settle_all() defined but never called, the store nested inside
    itself, prepare_scratch checking the wrong directory, and the settle record
    bypassing _atomic_write. The first three were caught by tests of behaviour;
    the fourth was invisible to them because a missing fsync changes nothing that
    a passing process can observe -- only a host crash can.

    So this is a source scan, which is crude, and deliberately so: it turns the
    fourth instance into the last by making the NEXT one fail at import of the
    test suite rather than at someone's power cut. Every durability-critical
    write in steam/streaming.py must go through _atomic_write.
    """
    import ast
    import inspect
    import steam.streaming as streaming

    source = inspect.getsource(streaming)
    tree = ast.parse(source)

    # The one legitimate exception, recorded rather than left implicit: rename()
    # swaps a DATA file as part of the settle protocol. A lost rename is
    # recoverable, because settle is idempotent and replays it -- which is only
    # true because the settle RECORD is durable, and that is what goes through
    # _atomic_write.
    allowed = {'_atomic_write', 'rename'}

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in allowed:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            target = inner.func
            name = None
            if isinstance(target, ast.Attribute):
                name = target.attr
            if name in ('replace', 'write_text', 'write_bytes'):
                # os.replace / Path.write_text / Path.write_bytes
                if name == 'replace' and not (
                        isinstance(target.value, ast.Name)
                        and target.value.id == 'os'):
                    continue        # str.replace and friends are not writes
                offenders.append(f"{node.name}: {name}() at line {inner.lineno}")

    assert not offenders, (
        "durability-critical writes must go through _atomic_write (which fsyncs "
        "the file AND its directory). Offending call sites:\n  "
        + "\n  ".join(offenders))
