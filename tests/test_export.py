import json
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from numpy.testing import assert_allclose
from yt.testing import fake_amr_ds
from yt.utilities.amr_kdtree.api import AMRKDTree

import yt_idv_export as standalone_export
from yt_idv_export import (
    assemble_scene,
    diagnose_export,
    export_block_scene,
    inspect_checkpoint,
)

LEFT, RIGHT = [0.25, 0.3, 0.35], [0.65, 0.7, 0.75]


@pytest.fixture()
def region():
    ds = fake_amr_ds()
    # the region holds only a weak reference to its dataset
    region = ds.region([0.5, 0.5, 0.5], LEFT, RIGHT)
    region._test_ds = ds
    return region


def _zip_contents(filename):
    with zipfile.ZipFile(filename) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


@pytest.mark.parametrize("no_ghost", [True, False])
def test_interrupted_export_resumes(region, tmp_path, monkeypatch, no_ghost):
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected, no_ghost=no_ghost)

    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    get_brick_data = AMRKDTree.get_brick_data
    calls = []

    def interrupt_on_third(self, node):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(node)
        return get_brick_data(self, node)

    monkeypatch.setattr(AMRKDTree, "get_brick_data", interrupt_on_third)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(
            region, "Density", filename, no_ghost=no_ghost, checkpoint_dir=checkpoint
        )
    assert not filename.exists()
    assert len(list((checkpoint / "blocks").glob("*.npz"))) == 2

    def count(self, node):
        calls.append(node)
        return get_brick_data(self, node)

    calls.clear()
    monkeypatch.setattr(AMRKDTree, "get_brick_data", count)
    export_block_scene(
        region, "Density", filename, no_ghost=no_ghost, checkpoint_dir=checkpoint
    )
    state = json.loads(_zip_contents(expected)["scene.json"])
    n_blocks = len(state["data_objects"][0]["attributes"]["block_data"]["__items__"])
    assert len(calls) == n_blocks - 2
    assert not checkpoint.exists()
    assert _zip_contents(filename) == _zip_contents(expected)


def test_checkpoint_from_another_export_is_rejected(region, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"

    def interrupt(self, node):
        raise KeyboardInterrupt

    monkeypatch.setattr(AMRKDTree, "get_brick_data", interrupt)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(
            region, "Density", tmp_path / "a.zip", checkpoint_dir=checkpoint
        )
    monkeypatch.undo()
    with pytest.raises(ValueError, match="different export .*no_ghost"):
        export_block_scene(
            region,
            "Density",
            tmp_path / "a.zip",
            no_ghost=True,
            checkpoint_dir=checkpoint,
        )


def test_checkpoint_assembles_without_dataset(region, tmp_path, monkeypatch):
    options = {
        "no_ghost": True,
        "float64": True,
        "float64_textures": True,
        "render_method": "projection",
    }
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected, **options)

    # the job ends after every block is saved, before the file is written
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setattr(standalone_export, "_write_scene", interrupt)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(
            region, "Density", filename, checkpoint_dir=checkpoint, **options
        )
    monkeypatch.undo()
    assert not filename.exists()

    moved = tmp_path / "elsewhere"
    shutil.copytree(checkpoint, moved)
    incomplete = tmp_path / "incomplete"
    shutil.copytree(checkpoint, incomplete)
    (incomplete / "blocks" / "000001.npz").unlink()
    with pytest.raises(ValueError, match="1 of the .* blocks"):
        assemble_scene(incomplete, filename)

    # with yt unimportable, as on a machine without it
    script = (
        "import runpy, sys; sys.modules['yt'] = None; "
        f"sys.argv = ['yt-idv-export', '--assemble', {str(moved)!r}, "
        f"{str(filename)!r}]; "
        f"runpy.run_path({standalone_export.__file__!r}, run_name='__main__')"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    assert moved.exists()
    assert _zip_contents(filename) == _zip_contents(expected)


class _FakeComm:
    # One rank of an MPI job, run one rank after another: the root rank records
    # what it broadcasts for the other ranks to replay, and its first barrier
    # runs the other ranks, which MPI would have run alongside it.
    comm = None

    def __init__(self, rank, size, broadcasts, at_barrier=None):
        self.rank = rank
        self.size = size
        self.broadcasts = broadcasts
        self.at_barrier = at_barrier

    def mpi_bcast(self, data, root=0):
        if self.rank == 0:
            self.broadcasts.append(data)
            return data
        return self.broadcasts.pop(0)

    def barrier(self):
        if self.at_barrier is not None:
            at_barrier, self.at_barrier = self.at_barrier, None
            at_barrier()


@pytest.mark.parametrize(
    ("no_ghost", "coarse_nodes"), [(True, False), (False, False), (False, True)]
)
def test_parallel_export(region, tmp_path, monkeypatch, no_ghost, coarse_nodes):
    options = {"no_ghost": no_ghost, "coarse_nodes": coarse_nodes}
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected, **options)

    filename = tmp_path / "scene.zip"
    size = 3
    comms = []
    calls = []
    get_brick_data = AMRKDTree.get_brick_data

    def count(self, node):
        calls.append((comms[-1].rank, node.node_id))
        return get_brick_data(self, node)

    monkeypatch.setattr(AMRKDTree, "get_brick_data", count)
    monkeypatch.setattr(standalone_export, "_communicator", lambda: comms[-1])
    grid_class = type(region.ds.index.grids[0])
    get_vertex_centered_data = grid_class.get_vertex_centered_data
    grids_read = []

    def count_grids(self, *args, **kwargs):
        grids_read.append(self.id)
        return get_vertex_centered_data(self, *args, **kwargs)

    monkeypatch.setattr(grid_class, "get_vertex_centered_data", count_grids)

    def run(comm):
        comms.append(comm)
        try:
            export_block_scene(region, "Density", filename, **options)
        finally:
            comms.pop()

    broadcasts = []

    def run_other_ranks():
        for rank in range(1, size):
            run(_FakeComm(rank, size, list(broadcasts)))
            # only the root rank writes the file
            assert not filename.exists()

    run(_FakeComm(0, size, broadcasts, run_other_ranks))

    state = json.loads(_zip_contents(expected)["scene.json"])
    attributes = state["data_objects"][0]["attributes"]
    n_blocks = len(attributes["block_data"]["__items__"])
    if coarse_nodes:
        n_blocks += len(attributes["_coarse_block_data"]["__items__"])
    assert len(calls) == len({node for _, node in calls}) == n_blocks
    assert {rank for rank, _ in calls} == set(range(size))
    # the ghost zones of each grid are made once, by the one rank given its blocks
    assert sorted(grids_read) == sorted(set(grids_read))
    assert not (tmp_path / "scene.zip.checkpoint").exists()
    assert _zip_contents(filename) == _zip_contents(expected)


def _n_blocks(filename):
    state = json.loads(_zip_contents(filename)["scene.json"])
    return len(state["data_objects"][0]["attributes"]["block_data"]["__items__"])


def test_export_without_assembling(region, tmp_path):
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected, no_ghost=True)

    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="checkpoint_dir"):
        export_block_scene(region, "Density", filename, assemble=False)
    export_block_scene(
        region,
        "Density",
        filename,
        no_ghost=True,
        checkpoint_dir=checkpoint,
        assemble=False,
    )
    assert not filename.exists()
    assert len(list((checkpoint / "blocks").glob("*.npz"))) == _n_blocks(expected)
    assemble_scene(checkpoint, filename)
    assert _zip_contents(filename) == _zip_contents(expected)


def _interrupt_on_third(monkeypatch, region, filename, checkpoint):
    get_brick_data = AMRKDTree.get_brick_data
    calls = []

    def interrupt_on_third(self, node):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(node)
        return get_brick_data(self, node)

    monkeypatch.setattr(AMRKDTree, "get_brick_data", interrupt_on_third)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(
            region, "Density", filename, no_ghost=True, checkpoint_dir=checkpoint
        )
    monkeypatch.undo()


def test_inspect_checkpoint(region, tmp_path, monkeypatch, capsys):
    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    _interrupt_on_third(monkeypatch, region, filename, checkpoint)
    # as left by a rank killed while it saved a block
    (checkpoint / "blocks" / "000005.npz.tmp").write_bytes(b"partial")
    capsys.readouterr()

    inspect_checkpoint(checkpoint, verify=True)
    report = capsys.readouterr().out
    assert "blocks: 2 of " in report
    assert "1 partly written file(s)" in report
    assert "stopped by KeyboardInterrupt" in report
    assert re.search(r"unfinished: block \d+ \(grid", report)
    assert "2 are complete" in report

    export_block_scene(
        region,
        "Density",
        filename,
        no_ghost=True,
        checkpoint_dir=checkpoint,
        assemble=False,
    )
    assert "removed 1 partly written" in capsys.readouterr().err
    inspect_checkpoint(checkpoint)
    report = capsys.readouterr().out
    assert "0 missing" in report
    assert "partly written" not in report
    assert "unfinished" not in report


def test_dry_run(region, tmp_path, monkeypatch, capsys):
    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    diagnose_export(region, "Density", no_ghost=True, checkpoint_dir=checkpoint)
    report = capsys.readouterr().err
    assert "holds no checkpoint" in report
    assert not checkpoint.exists()

    _interrupt_on_third(monkeypatch, region, filename, checkpoint)
    blocks = sorted((checkpoint / "blocks").iterdir())
    capsys.readouterr()
    diagnose_export(
        region,
        "Density",
        no_ghost=True,
        checkpoint_dir=checkpoint,
        n_ranks=2,
        n_sample=3,
    )
    report = capsys.readouterr().err
    assert "can be resumed; it holds 2 of the" in report
    assert "with 2 rank(s)" in report
    assert "grids to generate; with 2 ranks" in report
    assert report.count("generated the ") == 3
    assert sorted((checkpoint / "blocks").iterdir()) == blocks

    diagnose_export(region, "Density", no_ghost=False, checkpoint_dir=checkpoint)
    assert "different export (differs in no_ghost" in capsys.readouterr().err


@pytest.mark.parametrize("float64", [False, True])
@pytest.mark.parametrize("float64_textures", [False, True])
def test_scene_file_layout(region, tmp_path, float64, float64_textures):
    filename = tmp_path / "scene.zip"
    export_block_scene(
        region,
        "Density",
        filename,
        no_ghost=True,
        float64=float64,
        float64_textures=float64_textures,
    )
    contents = _zip_contents(filename)
    state = json.loads(contents["scene.json"])
    assert state["format_version"] == standalone_export.FORMAT_VERSION
    assert state["camera"]["class"] == standalone_export._CAMERA_CLASS
    assert state["components"][0]["class"] == standalone_export._COMPONENT_CLASS
    data = state["data_objects"][0]
    assert data["class"] == standalone_export._DATA_CLASS
    assert tuple(data["traits"]["field"]) == ("stream", "Density")

    def load(ref):
        with zipfile.ZipFile(filename) as zf, zf.open(ref["__array__"]) as f:
            return np.load(f)

    dtype = np.float64 if float64 else np.float32
    attributes = data["attributes"]["vertex_array"]["__vertex_array__"]["attributes"]
    for attribute in attributes:
        if attribute["name"] in ("in_dx", "in_left_edge", "in_right_edge"):
            assert load(attribute["data"]).dtype == dtype

    texture_dtype = np.float64 if float64_textures else np.float32
    items = data["attributes"]["block_data"]["__items__"]
    assert len(items) > 0
    for _, ref in items:
        assert load(ref).dtype == texture_dtype


def _scene(filename):
    contents = _zip_contents(filename)
    state = json.loads(contents["scene.json"])

    def load(ref):
        with zipfile.ZipFile(filename) as zf, zf.open(ref["__array__"]) as f:
            return np.load(f)

    return state["data_objects"][0]["attributes"], load


def _walk(kd, coarse_block, i=0, parent=-1):
    # the coarser block of each node of the flattened kd tree, depth first, with
    # the coarser block covering it from above (or -1)
    yield i, coarse_block[i], parent
    if kd["left"][i] != -1:
        here = coarse_block[i] if coarse_block[i] != -1 else parent
        yield from _walk(kd, coarse_block, kd["left"][i], here)
        yield from _walk(kd, coarse_block, kd["right"][i], here)


@pytest.mark.parametrize("no_ghost", [True, False])
def test_coarse_nodes(region, tmp_path, no_ghost):
    plain = tmp_path / "plain.zip"
    export_block_scene(region, "Density", plain, no_ghost=no_ghost)
    filename = tmp_path / "scene.zip"
    export_block_scene(
        region, "Density", filename, no_ghost=no_ghost, coarse_nodes=True
    )

    # the leaves are written exactly as without the coarser blocks
    expected = _zip_contents(plain)
    contents = _zip_contents(filename)
    state = json.loads(contents.pop("scene.json"))
    attributes = state["data_objects"][0]["attributes"]
    new = (
        "_coarse_block_data",
        "_coarse_block_bitmaps",
        "_coarse_edges",
        "_kd_coarse_block",
    )
    extra = {attributes.pop(k)["__array__"] for k in new[2:]}
    for k in new[:2]:
        extra |= {ref["__array__"] for _, ref in attributes.pop(k)["__items__"]}
    assert state == json.loads(expected.pop("scene.json"))
    assert {k: v for k, v in contents.items() if k not in extra} == expected

    attributes, load = _scene(filename)
    kd = {k: load(v) for k, v in attributes["_kd_tree"].items()}
    coarse_block = load(attributes["_kd_coarse_block"])
    assert coarse_block.shape == kd["block"].shape
    edges = load(attributes["_coarse_edges"])
    leaf_edges = load(attributes["_unscaled_edges"])
    data = dict(attributes["_coarse_block_data"]["__items__"])
    bitmaps = dict(attributes["_coarse_block_bitmaps"]["__items__"])
    n_coarse = len(data)
    assert n_coarse > 0
    assert edges.shape == (3, n_coarse, 3)

    # each coarser block appears once on a walk from the root, at an internal
    # node, and covers the leaves below it with coarser cells
    seen = []
    for i, c, above in _walk(kd, coarse_block):
        if kd["block"][i] != -1:
            b = kd["block"][i]
            if above != -1:
                assert np.all(edges[0, above] <= leaf_edges[0, b] + 1e-12)
                assert np.all(edges[1, above] >= leaf_edges[1, b] - 1e-12)
                assert np.all(edges[2, above] >= leaf_edges[2, b] * (1 - 1e-9))
        if c != -1:
            assert kd["left"][i] != -1
            assert above == -1 or np.all(edges[2, c] < edges[2, above] * (1 - 1e-9))
            seen.append(int(c))
    assert seen == list(range(n_coarse))

    ratio = attributes["_unitary_per_code_length"]
    max_val = state["data_objects"][0]["traits"]["max_val"]
    tiles = AMRKDTree(region.ds, data_source=region)
    tiles.fields = region._determine_fields(["Density"])
    tiles.log_fields = [False]
    tiles.no_ghost = no_ghost
    coarse = standalone_export._coarse_nodes(tiles.tree.trunk)
    assert len(coarse) == n_coarse
    for j, node in enumerate(coarse):
        brick = tiles.get_brick_data(node)
        assert_allclose(edges[0, j], brick.LeftEdge * ratio)
        assert_allclose(edges[1, j], brick.RightEdge * ratio)
        assert np.array_equal(load(bitmaps[j]), brick.source_mask * 255)
        expected = np.abs(np.asarray(brick.my_data[0])) / max_val
        expected[expected == 0.0] = np.finfo(np.float32).tiny
        assert np.array_equal(load(data[j]), expected.astype("f4"))


def test_coarse_nodes_checkpoint(region, tmp_path, monkeypatch):
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected, coarse_nodes=True)

    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"
    get_brick_data = AMRKDTree.get_brick_data
    calls = []

    def interrupt_on_third(self, node):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(node)
        return get_brick_data(self, node)

    monkeypatch.setattr(AMRKDTree, "get_brick_data", interrupt_on_third)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(
            region, "Density", filename, coarse_nodes=True, checkpoint_dir=checkpoint
        )
    monkeypatch.undo()
    with pytest.raises(ValueError, match="different export .*coarse_nodes"):
        export_block_scene(region, "Density", filename, checkpoint_dir=checkpoint)
    export_block_scene(
        region,
        "Density",
        filename,
        coarse_nodes=True,
        checkpoint_dir=checkpoint,
        assemble=False,
    )
    assemble_scene(checkpoint, filename)
    assert _zip_contents(filename) == _zip_contents(expected)


def test_force_periodicity(tmp_path, monkeypatch):
    ds = fake_amr_ds()
    ds._periodicity = (False, False, False)
    # the ghost zones at the domain's boundaries need periodicity
    region = ds.region([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="non-periodic domain"):
        export_block_scene(region, "Density", tmp_path / "plain.zip")
    export_block_scene(region, "Density", tmp_path / "plain.zip", no_ghost=True)

    ds.force_periodicity()
    expected = tmp_path / "expected.zip"
    export_block_scene(region, "Density", expected)

    filename = tmp_path / "scene.zip"
    checkpoint = tmp_path / "checkpoint"

    def interrupt(self, node):
        raise KeyboardInterrupt

    monkeypatch.setattr(AMRKDTree, "get_brick_data", interrupt)
    with pytest.raises(KeyboardInterrupt):
        export_block_scene(region, "Density", filename, checkpoint_dir=checkpoint)
    monkeypatch.undo()
    ds.force_periodicity(False)
    with pytest.raises(ValueError, match="different export .*force_periodicity"):
        export_block_scene(region, "Density", filename, checkpoint_dir=checkpoint)
    ds.force_periodicity()
    export_block_scene(region, "Density", filename, checkpoint_dir=checkpoint)
    assert _zip_contents(filename) == _zip_contents(expected)
