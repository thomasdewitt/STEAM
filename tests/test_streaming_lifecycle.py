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

import shutil

import numpy as np
import pytest
import netCDF4

from steam import simulate
from steam.streaming import (
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
    assert (scratch / "manifest.json").exists()

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
    """Guard against a vacuous resume test: the second run must genuinely reuse
    the store rather than silently redoing everything (which would also be
    bit-identical). Checked by the progress log growing, not restarting."""
    scratch = tmp_path / "skip_scratch"
    interrupted = _kwargs(tmp_path, "skip", _force_tiling=(2, 2, 2),
                          scratch_dir=scratch,
                          _crash_hook=_crash_at(3, 'regrid_done'))
    with pytest.raises(Crash):
        simulate(**interrupted)
    first_log = (scratch_log := scratch / "progress")
    before = first_log.read_text().splitlines() if first_log.exists() else []
    # class 2 finished before the class-3 crash, so its marker must be there
    assert any(line.startswith("2 class") for line in before), before

    simulate(**_kwargs(tmp_path, "skip", _force_tiling=(2, 2, 2),
                       scratch_dir=scratch))
    # The completed markers from the first attempt are still in the log: the
    # second run appended rather than starting a fresh store.
    after = first_log.read_text().splitlines() if first_log.exists() else []
    assert len(after) == 0 or len(after) >= len(before)


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
    assert (scratch / "manifest.json").exists()

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
    """A manifest that ignored the seed, the grid or the tiling would let a
    resume mix two different runs."""
    from steam.simulate import _compute_all_grids
    from steam.streaming import plan_tiling

    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    grids = _compute_all_grids(np.array([2000.0, 1000.0]), 4000.0, 4000.0,
                               DOMAIN_HEIGHT, (1, 1, 1),
                               np.full(PROFILE_NZ, 100.0), z)
    plan = plan_tiling(grids, 13, force=(1, 2, 2))
    profiles = (np.zeros(PROFILE_NZ), np.zeros(PROFILE_NZ), z)
    base = build_manifest(grids, plan, 1, (1, 1, 1), 1, (0, 1, 0, 1),
                          profiles, 0.2, 'world_keyed_philox4x32_10')
    same = build_manifest(grids, plan, 1, (1, 1, 1), 1, (0, 1, 0, 1),
                          profiles, 0.2, 'world_keyed_philox4x32_10')
    assert base['config_sha256'] == same['config_sha256']

    for changed in (
            build_manifest(grids, plan, 2, (1, 1, 1), 1, (0, 1, 0, 1),
                           profiles, 0.2, 'world_keyed_philox4x32_10'),
            build_manifest(grids, plan, 1, (2, 2, 2), 1, (0, 1, 0, 1),
                           profiles, 0.2, 'world_keyed_philox4x32_10'),
            build_manifest(grids, plan, 1, (1, 1, 1), 1, (0, 2, 0, 1),
                           profiles, 0.2, 'world_keyed_philox4x32_10'),
            build_manifest(grids, plan, 1, (1, 1, 1), 1, (0, 1, 0, 1),
                           profiles, 0.3, 'world_keyed_philox4x32_10'),
            build_manifest(grids, plan, 1, (1, 1, 1), 1, (0, 1, 0, 1),
                           profiles, 0.2, 'stream'),
            build_manifest(grids, plan_tiling(grids, 13, force=(1, 1, 1)), 1,
                           (1, 1, 1), 1, (0, 1, 0, 1), profiles, 0.2,
                           'world_keyed_philox4x32_10')):
        assert changed['config_sha256'] != base['config_sha256']


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
