"""
Export a yt data source as a scene that ``yt_idv.SceneGraph.load`` can open.

This file only depends on yt and numpy, so it can be copied to and run on a
machine where yt_idv (or OpenGL) is not available:

    yt-idv-export DATASET FIELD OUTPUT [--no-ghost] [--float64]
        [--checkpoint-dir DIR]

or, from python:

    from yt_idv_export import export_block_scene
    export_block_scene(ds.all_data(), "density", "scene.zip")

Then, wherever yt_idv is available:

    rc = yt_idv.render_context(height=800, width=800)
    rc.scene = yt_idv.scene_graph.SceneGraph.load("scene.zip")
    rc.run()

Only cartesian geometries are supported. The block processing mirrors
``BlockCollection.add_data`` and the file layout mirrors
``yt_idv.serialization``.

With ``--float64`` the vertex geometry and texture data are stored in float64;
yt_idv casts them down to float32 only for the GPU. That keeps deep zooms into
highly refined data exact, at the cost of a larger file (the textures double in
size), and needs a yt_idv that can load float64 scenes.

With ``--checkpoint-dir`` each block is saved to that directory as soon as yt
has generated it, so an export that is killed (e.g., by a batch job's time
limit) can be resumed by running the same command again; only the blocks that
were not saved yet are generated. The directory is removed once the scene file
has been written. Everything the scene file needs from the dataset is saved to
the directory before the first block, so once every block has been saved the
scene file can also be written from the directory alone, with numpy but
without yt or the dataset (e.g., on another machine, if the job ended before it
could write the file):

    yt-idv-export --assemble CHECKPOINT_DIR OUTPUT

The blocks can be generated in parallel with MPI, using yt's parallelism:

    mpirun -np 16 yt-idv-export DATASET FIELD OUTPUT --parallel \
        [--checkpoint-dir DIR]

(or ``yt.enable_parallelism()`` before calling ``export_block_scene``). Each
rank generates its share of the blocks and saves them to the checkpoint
directory, which must be on a filesystem that every rank can see; the root
rank then writes the scene file. A parallel export always checkpoints, to
``OUTPUT.checkpoint`` if no directory is given, and it can be resumed with any
number of ranks, or in serial. With ``--no-assemble`` the export stops once
every block is saved, so that a batch job does not spend its time writing the
scene file, which ``--assemble`` can do afterwards.

An export reports its progress on stderr: the blocks it will generate, what
each rank is working on every ``--log-interval`` seconds, and how long each
phase took. Each rank also appends the start and end of every block to a file
in the checkpoint directory. Two options help to find out why an export is
slow, or seems to do nothing:

    yt-idv-export DATASET FIELD OUTPUT --dry-run \
        [--checkpoint-dir DIR] [--ranks N] [--sample K]

reads only the dataset's index and reports how many blocks there are, how big
the checkpoint and the scene file will be, how much of the checkpoint can be
reused, and how the blocks would be divided among N ranks; with ``--sample``
it also generates K blocks (without saving them) to estimate how long the
export will take. And

    yt-idv-export --inspect CHECKPOINT_DIR [--verify]

needs neither yt nor the dataset, and can run while the export does: it
reports which blocks are saved, when they were saved, and what each rank was
doing when it last reported; ``--verify`` also reads every block file.
"""

import argparse
import contextlib
import datetime
import json
import os
import shutil
import socket
import sys
import threading
import time
import zipfile

import numpy as np

FORMAT_VERSION = 1

_CAMERA_CLASS = "yt_idv.cameras.trackball_camera.TrackballCamera"
_DATA_CLASS = "yt_idv.scene_data.block_collection.BlockCollection"
_COMPONENT_CLASS = "yt_idv.scene_components.blocks.BlockRendering"
_TEXTURE_CLASS = "yt_idv.opengl_support.Texture3D"


def dataset_info(ds):
    """yt_idv.serialization.dataset_info, repeated so this module needs no OpenGL."""
    return {
        "unit_registry": ds.unit_registry.to_json(),
        "current_redshift": float(getattr(ds, "current_redshift", 0.0) or 0.0),
        "current_time": str(getattr(ds, "current_time", "")),
    }


def _log(message, rank=None):
    # to stderr and flushed, so that a batch job's log is up to date
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    who = "" if rank is None else f" [rank {rank}]"
    print(f"{stamp}{who} {message}", file=sys.stderr, flush=True)


def _format_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _format_seconds(seconds):
    if not np.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _format_ranges(indices, limit=20):
    ranges = []
    for i in indices:
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i, i])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges[:limit])
    if len(ranges) > limit:
        text += f", ... ({len(ranges) - limit} more runs)"
    return text


class _Progress:
    # Logs how far along the work is every `interval` seconds from a thread of
    # its own, so that a block that takes long (or never finishes) still shows
    # up in the log. With a `record` file, also appends when each block starts
    # and ends, for inspect_checkpoint.
    def __init__(self, what, total, rank=None, interval=300, record=None, run=None):
        self.what = what
        self.run = run
        self.total = total
        self.rank = rank
        self.interval = interval
        self.record = record
        self.n_done = 0
        self.current = None
        self.start = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def __enter__(self):
        if self.record is not None:
            os.makedirs(os.path.dirname(self.record), exist_ok=True)
        self._record(
            f"begin run={self.run} host={socket.gethostname()} pid={os.getpid()} "
            f"blocks={self.total}"
        )
        if self.interval:
            threading.Thread(target=self._heartbeat, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        ended = "finished" if exc_type is None else f"stopped by {exc_type.__name__}"
        self._record(f"end {ended}")
        _log(f"{self.what}: {ended}; {self.status()}", self.rank)

    def _record(self, line):
        if self.record is None:
            return
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.record, "a") as f:
            f.write(f"{stamp} {line}\n")

    def begin(self, i, about=""):
        with self._lock:
            self.current = (i, about, time.monotonic())
        self._record(f"start {i} {about}".rstrip())

    def end(self, i):
        with self._lock:
            elapsed = time.monotonic() - self.current[2]
            self.current = None
            self.n_done += 1
        self._record(f"done {i} {elapsed:.2f}")

    def status(self):
        with self._lock:
            n_done, current = self.n_done, self.current
        elapsed = time.monotonic() - self.start
        message = (
            f"{n_done}/{self.total} ({100 * n_done / max(self.total, 1):.1f}%) "
            f"in {_format_seconds(elapsed)}"
        )
        if 0 < n_done < self.total:
            remaining = elapsed / n_done * (self.total - n_done)
            message += f", about {_format_seconds(remaining)} to go"
        if current is not None:
            i, about, started = current
            message += (
                f"; on block {i} {about} for "
                f"{_format_seconds(time.monotonic() - started)}"
            )
        return message

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            _log(f"{self.what}: {self.status()}", self.rank)


class _Archive:
    # Arrays are written to the zip file as they are added, so they are not
    # all held in memory at once.
    def __init__(self, zf):
        self.zf = zf
        self.n_arrays = 0
        self.textures = []

    def add_array(self, arr):
        key = f"arrays/{self.n_arrays:06d}.npy"
        self.n_arrays += 1
        with self.zf.open(key, "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)
        return {"__array__": key}

    def add_texture(self, data, **parameters):
        self.textures.append(
            {
                "class": _TEXTURE_CLASS,
                "parameters": parameters,
                "data": self.add_array(data),
            }
        )
        return {"__texture__": len(self.textures) - 1}

    def write_state(self, state):
        self.zf.writestr("scene.json", json.dumps(state, indent=2))


def _flatten_kd_tree(trunk):
    nodes = list(trunk.depth_traverse())
    index = {id(node): i for i, node in enumerate(nodes)}
    n = len(nodes)
    left = np.full(n, -1, dtype="int64")
    right = np.full(n, -1, dtype="int64")
    split_dim = np.full(n, -1, dtype="int64")
    split_pos = np.full(n, np.nan, dtype="float64")
    block = np.full(n, -1, dtype="int64")
    n_blocks = 0
    for i, node in enumerate(nodes):
        if node.kd_is_leaf():
            if node.grid != -1:
                block[i] = n_blocks
                n_blocks += 1
            continue
        left[i] = index[id(node.left)]
        right[i] = index[id(node.right)]
        split_dim[i] = node.get_split_dim()
        split_pos[i] = node.get_split_pos()
    return {
        "left": left,
        "right": right,
        "split_dim": split_dim,
        "split_pos": split_pos,
        "block": block,
    }


class _MemoryStore:
    def __init__(self):
        self._store = {}

    def __contains__(self, key):
        return key in self._store

    def saved(self):
        return set(self._store)

    def progress_fn(self, rank):
        return None

    def save(self, i, record):
        self._store[i] = record

    def load(self, i, keys):
        return {k: self._store[i][k] for k in keys}


class _CheckpointStore:
    # Each block is written to a temporary file and then renamed, so a block
    # file is either complete or absent no matter when the export is killed.
    def __init__(self, path, manifest=None, tree=None):
        self.path = path
        self.resumed = True
        if manifest is None:
            if not os.path.exists(os.path.join(path, "manifest.json")):
                raise ValueError(f"{path} does not hold a checkpoint.")
            return
        mismatch = self.mismatch(path, manifest, tree)
        if mismatch:
            raise ValueError(
                f"The checkpoint in {path} was made for a different export "
                f"({mismatch}); remove it to start over."
            )
        if os.path.exists(os.path.join(path, "manifest.json")):
            return
        self.resumed = False
        tree_fn = os.path.join(path, "tree.npz")
        manifest_fn = os.path.join(path, "manifest.json")
        os.makedirs(os.path.join(path, "blocks"), exist_ok=True)
        self._write(tree_fn, lambda f: np.savez(f, **tree))
        self._write(manifest_fn, lambda f: f.write(json.dumps(manifest).encode()))

    @staticmethod
    def mismatch(path, manifest, tree):
        """How a checkpoint in path differs from an export's, or "" if not."""
        manifest_fn = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_fn):
            return ""
        with open(manifest_fn) as f:
            saved = json.load(f)
        differences = [
            k
            for k in sorted(set(saved) | set(manifest))
            if saved.get(k) != manifest.get(k)
        ]
        with np.load(os.path.join(path, "tree.npz")) as f:
            if set(f.files) != set(tree) or not all(
                np.array_equal(f[k], v, equal_nan=True) for k, v in tree.items()
            ):
                differences.append("kd tree")
        return "" if not differences else "differs in " + ", ".join(differences)

    @staticmethod
    def _write(fn, write):
        with open(f"{fn}.tmp", "wb") as f:
            write(f)
        os.replace(f"{fn}.tmp", fn)

    def _scene_fn(self):
        return os.path.join(self.path, "scene.json")

    def save_scene(self, scene):
        self._write(self._scene_fn(), lambda f: f.write(json.dumps(scene).encode()))

    def load_scene(self):
        with open(self._scene_fn()) as f:
            return json.load(f)

    def load_tree(self):
        with np.load(os.path.join(self.path, "tree.npz"), allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    def _block_fn(self, i):
        return os.path.join(self.path, "blocks", f"{i:06d}.npz")

    def __contains__(self, i):
        return os.path.exists(self._block_fn(i))

    def _scan_blocks(self):
        # one listing of the directory, which on a parallel filesystem is much
        # faster than checking for each block file in turn
        saved, temporary = {}, []
        for entry in os.scandir(os.path.join(self.path, "blocks")):
            stem = entry.name.removesuffix(".npz")
            if entry.name.endswith(".tmp"):
                temporary.append(entry)
            elif entry.name.endswith(".npz") and stem.isdigit():
                saved[int(stem)] = entry
        return saved, temporary

    def saved(self):
        return set(self._scan_blocks()[0])

    def remove_temporary(self):
        # left by blocks whose writing was cut short; only safe to call while
        # no rank is generating blocks
        temporary = self._scan_blocks()[1]
        for entry in temporary:
            os.remove(entry.path)
        return len(temporary)

    def progress_fn(self, rank):
        return os.path.join(self.path, "progress", f"rank_{rank:05d}.log")

    def save(self, i, record):
        self._write(self._block_fn(i), lambda f: np.savez(f, **record))

    def load(self, i, keys):
        with np.load(self._block_fn(i), allow_pickle=False) as f:
            return {k: f[k] for k in keys}


def _communicator():
    from yt.utilities.parallel_tools.parallel_analysis_interface import (
        communication_system,
    )

    return communication_system.communicators[-1]


@contextlib.contextmanager
def _rank_local(comm):
    # yt divides its work among the ranks of the innermost communicator: an
    # AMRKDTree gives each rank a different part of the domain, and data objects
    # made while generating a block (e.g., for ghost zones) reduce over all the
    # ranks. Here every rank needs the whole tree and generates its blocks on
    # its own, so yt is handed a communicator of this rank alone.
    if comm.comm is None or comm.comm.size == 1:
        yield
        return
    from mpi4py import MPI
    from yt.utilities.parallel_tools.parallel_analysis_interface import (
        communication_system,
    )

    # a duplicate, as yt frees its communicators
    communication_system.push(MPI.COMM_SELF.Dup())
    try:
        yield
    finally:
        communication_system.pop()


def _block_shape(tiles, node):
    grid = tiles.ds.index.grids[node.grid - tiles._id_offset]
    width = node.get_right_edge() - node.get_left_edge()
    return grid, np.rint(width / grid.dds.ndarray_view()).astype("int64")


def _describe_block(grid, dims):
    return f"(grid {grid.id}, level {grid.Level}, {'x'.join(map(str, dims))} cells)"


def _block_sizes(tiles, no_ghost):
    """The level, cell count and data size (with ghost zones) of each block."""
    levels, cells, values, grids = [], [], [], set()
    for node in tiles.tree.trunk.kd_traverse():
        grid, dims = _block_shape(tiles, node)
        grids.add(grid.id)
        levels.append(grid.Level)
        cells.append(dims.prod())
        # with ghost zones, the data is vertex centered
        values.append((dims + (0 if no_ghost else 1)).prod())
    return np.array(levels), np.array(cells), np.array(values), len(grids)


def _summarize_blocks(tiles, no_ghost, float64):
    levels, cells, values, n_grids = _block_sizes(tiles, no_ghost)
    lines = [
        f"{n_grids} of the dataset's {len(tiles.ds.index.grids)} grids are "
        f"split into {len(levels)} blocks by the kd tree "
        f"({cells.sum():,} cells); each block is a file in the checkpoint",
    ]
    for level in np.unique(levels):
        on_level = levels == level
        lines.append(
            f"  level {level}: {on_level.sum()} blocks, "
            f"{cells[on_level].sum():,} cells, "
            f"the largest {cells[on_level].max():,}"
        )
    lines.append(
        f"expected size: about {_format_bytes(values.sum() * 8 + cells.sum())} "
        f"of checkpoint, {_format_bytes(values.sum() * (8 if float64 else 4))} "
        "of scene file before any compression"
    )
    return lines, values


def _by_grid(blocks, nodes):
    """The blocks grouped by their grid, in the order of each grid's first."""
    groups = {}
    for i in blocks:
        groups.setdefault(nodes[i].grid, []).append(i)
    return list(groups.values())


def _forget_grids(tiles):
    # AMRKDTree keeps the data (with ghost zones) of every grid it has read,
    # for its other blocks, and never lets go of it. The blocks are generated
    # one grid at a time, so once a grid's blocks are done its data can go.
    tiles.current_saved_grids = []
    tiles.current_vcds = []


def _generate_blocks(tiles, nodes, store, groups, comm, log_interval, run):
    from yt.funcs import get_pbar

    n_blocks = sum(len(group) for group in groups)
    pbar = get_pbar("Generating blocks", n_blocks)
    progress = _Progress(
        "generating blocks",
        n_blocks,
        rank=comm.rank if comm.size > 1 else None,
        interval=log_interval,
        record=store.progress_fn(comm.rank),
        run=run,
    )
    with progress:
        try:
            for group in groups:
                for i in group:
                    node = nodes[i]
                    progress.begin(i, _describe_block(*_block_shape(tiles, node)))
                    with _rank_local(comm):
                        brick = tiles.get_brick_data(node)
                    data = brick.my_data[0]
                    abs_data = np.abs(data)
                    store.save(
                        i,
                        {
                            "data": np.asarray(data),
                            "mask": brick.source_mask,
                            "left_edge": brick.LeftEdge,
                            "right_edge": brick.RightEdge,
                            "dx": (brick.RightEdge - brick.LeftEdge)
                            / brick.source_mask.shape,
                            "min_val": np.asarray(np.nanmin(abs_data)),
                            "max_val": np.asarray(np.nanmax(abs_data)),
                            "units": np.array(str(getattr(data, "units", ""))),
                        },
                    )
                    node.data = None
                    progress.end(i)
                    pbar.update(progress.n_done)
                _forget_grids(tiles)
        finally:
            pbar.finish()


def _camera_state(ds):
    center = ds.domain_center.to("unitary").d
    position = center + 1.5 * ds.domain_width.to("unitary").d
    near_plane = 3.0 * ds.index.get_smallest_dx().min().to("unitary").d
    return {
        "position": position.tolist(),
        "focus": center.tolist(),
        "up": [0.0, 1.0, 0.0],
        "near_plane": max(float(near_plane), 1e-5),
    }


def export_block_scene(
    data_source,
    field,
    filename,
    no_ghost=False,
    render_method=None,
    compress=False,
    float64=False,
    checkpoint_dir=None,
    assemble=True,
    log_interval=300,
):
    """
    Write a volume rendering of a data source to a yt_idv scene file.

    Parameters
    ----------
    data_source: yt data container such as a sphere, region, etc
    field: The field to volume render
    filename: The file to write
    no_ghost: Should we save time by skipping ghost zone generation
    render_method: The BlockRendering render method (e.g., "projection");
                   uses the yt_idv default if None
    compress: Should the file be compressed (slower, but often smaller)
    float64: Store the vertex geometry and texture data in float64 (yt_idv
             casts them to float32 for the GPU); otherwise float32
    checkpoint_dir: A directory to save each block to as it is generated, so
                    that an interrupted export can be resumed by calling this
                    again with the same arguments; removed once the file has
                    been written. With yt's parallelism enabled, the blocks are
                    divided among the MPI ranks, which save them here (default
                    "<filename>.checkpoint"), and the root rank writes the file
    assemble: Write the scene file once every block is saved; if False, stop
              there and leave the blocks in checkpoint_dir for assemble_scene
    log_interval: Seconds between progress reports on stderr (0 for none);
                  the start and end of each phase are always reported
    """
    comm = _communicator()
    root = comm.rank == 0
    if not assemble and checkpoint_dir is None and comm.size == 1:
        raise ValueError("Without assembling, the blocks need a checkpoint_dir.")
    started = time.monotonic()
    if root:
        _log(f"exporting {field} of {data_source} with {comm.size} rank(s)")
    tiles, field, tree, bbox, manifest = _setup(data_source, field, no_ghost, comm)
    ds = data_source.ds
    if root:
        _log(f"built the kd tree in {_format_seconds(time.monotonic() - started)}")
        for line in _summarize_blocks(tiles, no_ghost, float64)[0]:
            _log(line)
    if checkpoint_dir is None and comm.size > 1:
        checkpoint_dir = f"{filename}.checkpoint"
    if checkpoint_dir is None:
        store = _MemoryStore()
    else:
        # some datasets, e.g. in-memory ones, get a different identifier in
        # each process
        manifest = comm.mpi_bcast(manifest)
        # the root rank sets up the directory before the others look at it
        error = None
        if root:
            try:
                store = _CheckpointStore(checkpoint_dir, manifest, tree)
            except ValueError as e:
                error = str(e)
        error = comm.mpi_bcast(error)
        if error is not None:
            raise ValueError(error)
        if not root:
            store = _CheckpointStore(checkpoint_dir, manifest, tree)
    scene = None
    if root:
        # all that the scene file needs from the dataset, saved before any
        # block so that the file can be written without it
        scene = {
            "geometry": str(ds.geometry),
            "field": list(field),
            "bbox": bbox.tolist(),
            "unitary_per_code_length": float(
                (ds.units.code_length / ds.units.unitary).base_value
            ),
            "camera": _camera_state(ds),
            "dataset": dataset_info(ds),
            "render_method": render_method,
            "compress": bool(compress),
            "float64": bool(float64),
        }
        if checkpoint_dir is not None:
            store.save_scene(scene)
    n_blocks = int((tree["block"] >= 0).sum())
    missing = None
    # names this run in the progress records, to tell it from earlier ones
    run = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if root:
        if checkpoint_dir is not None:
            _log_checkpoint(store, checkpoint_dir, n_blocks)
        missing = sorted(set(range(n_blocks)) - store.saved())
    run, missing = comm.mpi_bcast((run, missing))
    nodes = list(tiles.tree.trunk.kd_traverse())
    groups = _by_grid(missing, nodes)
    if root:
        _log(_plan(groups, comm.size))
    _generate_blocks(
        tiles,
        nodes,
        store,
        groups[comm.rank :: comm.size],
        comm,
        log_interval,
        run,
    )
    waiting = time.monotonic()
    comm.barrier()
    if root and comm.size > 1:
        _log(
            "every rank has finished its blocks; waited "
            f"{_format_seconds(time.monotonic() - waiting)} for the others"
        )
    if not assemble:
        if root:
            _log(
                f"all {n_blocks} blocks are saved in {checkpoint_dir}; write the "
                "scene file with: yt-idv-export --assemble "
                f"{checkpoint_dir} {filename}"
            )
        return
    if root:
        _write_scene(store, tree, scene, filename, log_interval)
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir)
            _log(f"removed {checkpoint_dir}")
        _log(f"export finished in {_format_seconds(time.monotonic() - started)}")
    elif comm.rank == 1:
        _log("waiting for the root rank to write the scene file", comm.rank)
    # every rank returns once the file exists
    comm.barrier()


def _setup(data_source, field, no_ghost, comm):
    ds = data_source.ds
    geometry = str(ds.geometry)
    if geometry != "cartesian":
        raise NotImplementedError(
            f"Only cartesian geometries can be exported, not {geometry}."
        )

    from yt.utilities.amr_kdtree.api import AMRKDTree

    # AMRKDTree.set_fields would generate every block in one call, which can
    # not be checkpointed, so the blocks are generated one at a time instead.
    # A tree of our own also keeps the data source's cached tree untouched.
    with _rank_local(comm):
        tiles = AMRKDTree(ds, data_source=data_source)
    tiles.fields = data_source._determine_fields([field])
    tiles.log_fields = [False]
    tiles.no_ghost = no_ghost
    field = tiles.fields[0]

    tree = _flatten_kd_tree(tiles.tree.trunk)
    bbox = np.array([edge.to("unitary").d for edge in data_source.get_bbox()])
    manifest = {
        "dataset": str(ds),
        "unique_identifier": str(getattr(ds, "unique_identifier", "")),
        "field": list(field),
        "no_ghost": bool(no_ghost),
        "bbox": bbox.tolist(),
    }
    return tiles, field, tree, bbox, manifest


def _plan(groups, n_ranks):
    # Each grid's blocks go to one rank, as most of the time a block takes goes
    # into the ghost zones of its whole grid, which the grid's other blocks
    # reuse. The grids are dealt out, as neighboring grids in the tree tend to
    # be similarly refined and so similarly costly.
    n_blocks = sum(len(group) for group in groups)
    plan = f"{n_blocks} blocks of {len(groups)} grids to generate"
    if n_ranks > 1:
        shares = [groups[r::n_ranks] for r in range(n_ranks)]
        per_rank = [sum(len(group) for group in share) for share in shares]
        plan += (
            f"; with {n_ranks} ranks, {len(shares[-1])} to {len(shares[0])} grids "
            f"and {min(per_rank)} to {max(per_rank)} blocks per rank"
        )
    return plan


def _log_checkpoint(store, checkpoint_dir, n_blocks):
    if not store.resumed:
        _log(f"started a new checkpoint in {checkpoint_dir}")
        return
    saved = store.saved()
    _log(
        f"resuming from the checkpoint in {checkpoint_dir}, which holds "
        f"{len(saved & set(range(n_blocks)))} of the {n_blocks} blocks"
    )
    n_temporary = store.remove_temporary()
    if n_temporary:
        _log(f"removed {n_temporary} partly written block file(s)")
    stray = sorted(saved - set(range(n_blocks)))
    if stray:
        _log(f"warning: ignoring block files beyond the last block: {stray[:10]}")


def _write_scene(store, tree, scene, filename, log_interval=300):
    # Needs only numpy, so a checkpoint can be assembled on any machine.
    n_blocks = int((tree["block"] >= 0).sum())
    started = time.monotonic()
    _log(
        f"writing {filename} from {n_blocks} blocks (float64={scene['float64']}, "
        f"compress={scene['compress']})"
    )
    keys = ("left_edge", "right_edge", "dx", "min_val", "max_val")
    summary = []
    with _Progress(
        "reading the blocks' extents", n_blocks, interval=log_interval
    ) as progress:
        for i in range(n_blocks):
            progress.begin(i)
            summary.append(store.load(i, keys))
            progress.end(i)
    field_units = str(store.load(0, ("units",))["units"])
    max_val = float(max(b["max_val"] for b in summary))

    dx = np.array([b["dx"] for b in summary])
    le = np.array([b["left_edge"] for b in summary])
    re = np.array([b["right_edge"] for b in summary])
    diagonal = float(np.sqrt(((re.max(axis=0) - le.min(axis=0)) ** 2).sum()))
    ratio = scene["unitary_per_code_length"]
    bbox = np.array(scene["bbox"])
    dtype = "f8" if scene["float64"] else "f4"
    render_method = scene["render_method"]

    mode = zipfile.ZIP_DEFLATED if scene["compress"] else zipfile.ZIP_STORED
    partial = f"{filename}.partial"
    try:
        with zipfile.ZipFile(partial, "w", compression=mode, allowZip64=True) as zf:
            archive = _Archive(zf)
            vertex_array = {
                "name": "block_info",
                "each": 1,
                "indices": None,
                "attributes": [
                    {
                        "name": name,
                        "divisor": 0,
                        "data": archive.add_array(arr),
                    }
                    for name, arr in (
                        ("model_vertex", np.ones((n_blocks, 4), dtype="f4")),
                        ("in_dx", (dx * ratio).astype(dtype)),
                        ("in_left_edge", (le * ratio).astype(dtype)),
                        ("in_right_edge", (re * ratio).astype(dtype)),
                    )
                ],
            }

            block_data = []
            block_bitmaps = []
            progress = _Progress(
                f"writing blocks to {partial}", n_blocks, interval=log_interval
            )
            with progress:
                for i in range(n_blocks):
                    progress.begin(i)
                    block = store.load(i, ("data", "mask"))
                    n_data = np.abs(block["data"]).astype("f8")
                    if max_val > 0:
                        # Divided by the maximum alone: subtracting the minimum
                        # collapses values just above it toward 0, which a log
                        # transfer function spreads over many decades.
                        n_data /= max_val
                        # zero-valued blocks are skipped by the shader
                        n_data[n_data == 0.0] = np.finfo(np.float32).tiny
                    n_data = n_data.astype(dtype)
                    block_data.append([i, archive.add_array(n_data)])
                    block_bitmaps.append(
                        [i, archive.add_array((block["mask"] * 255).astype("uint8"))]
                    )
                    progress.end(i)

            block_collection = {
                "class": _DATA_CLASS,
                "traits": {
                    # the data is normalized by max_val alone
                    "min_val": 0.0,
                    "max_val": max_val,
                    "field": scene["field"],
                    "field_units": field_units,
                    "_yt_geom_str": scene["geometry"],
                },
                "attributes": {
                    "vertex_array": {"__vertex_array__": vertex_array},
                    "textures": [],
                    "block_data": {"__items__": block_data},
                    "block_bitmaps": {"__items__": block_bitmaps},
                    "_kd_tree": {k: archive.add_array(v) for k, v in tree.items()},
                    "diagonal": diagonal,
                    # the data source's bounding box (unitary), BlockCollection.bbox
                    "_bbox": archive.add_array(bbox),
                    # float64 unscaled edges (unitary) and the unit conversion, so the
                    # scene can be scaled and rescaled without losing precision
                    "_unscaled_edges": archive.add_array(
                        np.array([le * ratio, re * ratio, dx * ratio])
                    ),
                    "_unitary_per_code_length": ratio,
                },
            }
            component = {
                "class": _COMPONENT_CLASS,
                "traits": {}
                if render_method is None
                else {"render_method": render_method},
                "attributes": {},
                "data": {"__data__": 0},
            }
            state = {
                "format_version": FORMAT_VERSION,
                "camera": {"class": _CAMERA_CLASS, "state": scene["camera"]},
                "n_scene_data_objects": 1,
                "components": [component],
                "annotations": [],
                "data_objects": [block_collection],
                "textures": archive.textures,
                "dataset": scene["dataset"],
            }
            archive.write_state(state)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, filename)
    _log(
        f"wrote {filename} ({_format_bytes(os.path.getsize(filename))}) in "
        f"{_format_seconds(time.monotonic() - started)}"
    )


def assemble_scene(checkpoint_dir, filename, log_interval=300):
    """
    Write the scene file of an export whose blocks were all saved to
    ``checkpoint_dir``, without yt or the dataset (e.g., on another machine,
    when the export ran out of time before it wrote the file). The scene is
    written with the options of the export, and the checkpoint is left as is.

    Parameters
    ----------
    checkpoint_dir: The checkpoint directory of the export
    filename: The file to write
    log_interval: Seconds between progress reports on stderr (0 for none)
    """
    store = _CheckpointStore(checkpoint_dir)
    try:
        scene = store.load_scene()
    except FileNotFoundError:
        raise ValueError(
            f"{checkpoint_dir} has no scene.json, as it was started by an older "
            "version of this script; run the export once more to add it."
        ) from None
    tree = store.load_tree()
    n_blocks = int((tree["block"] >= 0).sum())
    n_missing = len(set(range(n_blocks)) - store.saved())
    if n_missing:
        raise ValueError(
            f"{n_missing} of the {n_blocks} blocks in {checkpoint_dir} have not "
            "been generated yet; run the export again to generate them."
        )
    _write_scene(store, tree, scene, filename, log_interval)


def diagnose_export(
    data_source,
    field,
    no_ghost=False,
    float64=False,
    checkpoint_dir=None,
    n_ranks=1,
    n_sample=0,
):
    """
    Report on stderr what an export would do, without generating or saving
    any blocks: how the data source is split into blocks, how big the
    checkpoint and the scene file would be, how much of ``checkpoint_dir``
    could be reused, and how the remaining blocks would be divided among
    ``n_ranks`` ranks. Only the dataset's index is read, unless ``n_sample``
    blocks are generated (but not saved) to estimate how long the export
    would take.

    Parameters
    ----------
    data_source: yt data container such as a sphere, region, etc
    field: The field to volume render
    no_ghost: As for export_block_scene
    float64: As for export_block_scene
    checkpoint_dir: The checkpoint directory the export would use, if any
    n_ranks: The number of MPI ranks to plan for
    n_sample: The number of blocks to generate to time them
    """
    started = time.monotonic()
    _log(f"building the kd tree for {field} of {data_source}")
    tiles, field, tree, bbox, manifest = _setup(
        data_source, field, no_ghost, _communicator()
    )
    _log(f"built the kd tree in {_format_seconds(time.monotonic() - started)}")
    lines, values = _summarize_blocks(tiles, no_ghost, float64)
    for line in lines:
        _log(line)
    n_blocks = len(values)
    missing = list(range(n_blocks))
    if checkpoint_dir is not None:
        if not os.path.exists(os.path.join(checkpoint_dir, "manifest.json")):
            _log(f"{checkpoint_dir} holds no checkpoint; the export would start one")
        elif mismatch := _CheckpointStore.mismatch(checkpoint_dir, manifest, tree):
            _log(
                f"the checkpoint in {checkpoint_dir} can not be resumed, as it "
                f"was made for a different export ({mismatch})"
            )
        else:
            missing = sorted(set(missing) - _CheckpointStore(checkpoint_dir).saved())
            _log(
                f"the checkpoint in {checkpoint_dir} can be resumed; it holds "
                f"{n_blocks - len(missing)} of the {n_blocks} blocks"
            )
    nodes = list(tiles.tree.trunk.kd_traverse())
    groups = _by_grid(missing, nodes)
    _log(_plan(groups, n_ranks))
    if not (n_sample and groups):
        return
    sample = sorted({int(j) for j in np.linspace(0, len(groups) - 1, n_sample)})
    seconds = []
    for group in (groups[j] for j in sample):
        grid, _ = _block_shape(tiles, nodes[group[0]])
        start = time.monotonic()
        for i in group:
            tiles.get_brick_data(nodes[i])
            nodes[i].data = None
        _forget_grids(tiles)
        seconds.append(time.monotonic() - start)
        _log(
            f"generated the {len(group)} block(s) of grid {grid.id} (level "
            f"{grid.Level}, {'x'.join(map(str, grid.ActiveDimensions))} cells) "
            f"in {seconds[-1]:.2f} s"
        )
    per_grid = float(np.mean(seconds))
    most = -(-len(groups) // n_ranks)
    _log(
        f"at the sample's mean of {per_grid:.3g} s per grid, the blocks would "
        f"take about {_format_seconds(per_grid * most)} with {n_ranks} rank(s), "
        "if every rank reads as fast as this one did"
    )


def inspect_checkpoint(checkpoint_dir, verify=False):
    """
    Print a report on a checkpoint directory: which blocks are saved, when
    they were saved, what each rank was doing when it last reported, and
    (with ``verify``) whether every block file can be read. Needs only numpy,
    so it can run wherever the directory can be read, also while the export
    is running.

    Parameters
    ----------
    checkpoint_dir: The checkpoint directory of an export
    verify: Read every block file, to check that it is complete
    """
    store = _CheckpointStore(checkpoint_dir)
    now = time.time()
    with open(os.path.join(checkpoint_dir, "manifest.json")) as f:
        manifest = json.load(f)
    print(f"checkpoint {os.path.abspath(checkpoint_dir)}")
    for key, value in manifest.items():
        print(f"  {key}: {value}")
    try:
        scene = store.load_scene()
    except FileNotFoundError:
        print(
            "  no scene.json (started by an older version); run the export "
            "once more before --assemble"
        )
    else:
        print(
            f"  scene file options: float64={scene['float64']}, "
            f"compress={scene['compress']}, render_method={scene['render_method']}"
        )

    tree = store.load_tree()
    n_blocks = int((tree["block"] >= 0).sum())
    saved, temporary = store._scan_blocks()
    blocks = {i: entry for i, entry in saved.items() if i < n_blocks}
    missing = sorted(set(range(n_blocks)) - set(blocks))
    print()
    print(
        f"blocks: {len(blocks)} of {n_blocks} saved "
        f"({100 * len(blocks) / max(n_blocks, 1):.1f}%), {len(missing)} missing"
    )
    if missing:
        print(f"  missing: {_format_ranges(missing)}")
    stray = sorted(set(saved) - set(blocks))
    if stray:
        print(f"  {len(stray)} files beyond the last block: {_format_ranges(stray)}")
    if temporary:
        print(
            f"  {len(temporary)} partly written file(s), removed when the export "
            f"resumes: {', '.join(sorted(e.name for e in temporary)[:5])}"
        )
    if blocks:
        stats = [entry.stat() for entry in blocks.values()]
        sizes = np.array([st.st_size for st in stats])
        print(
            f"  {_format_bytes(sizes.sum())} in all, "
            f"{_format_bytes(sizes.mean())} per block on average, "
            f"the largest {_format_bytes(sizes.max())}"
        )
        _print_timeline(np.sort([st.st_mtime for st in stats]), now)
    _print_progress(_read_progress(checkpoint_dir), set(blocks), now)
    if verify:
        _verify_blocks(store, sorted(blocks))


def _format_time(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _print_timeline(mtimes, now):
    first, last = mtimes[0], mtimes[-1]
    print()
    print(
        f"blocks were saved from {_format_time(first)} to {_format_time(last)}; "
        f"the last one {_format_seconds(now - last)} ago"
    )
    bucket = 3600 * max(1, int(np.ceil((last - first) / 3600 / 48)))
    edges = np.arange(first - first % bucket, last + bucket, bucket)
    counts = np.histogram(mtimes, edges)[0]
    print(f"  blocks saved per {bucket // 3600} hour(s):")
    for start, count in zip(edges, counts, strict=False):
        bar = "#" * int(round(50 * count / counts.max()))
        print(f"  {_format_time(start)} {count:7d} {bar}")
    gaps = np.diff(mtimes)
    longest = [j for j in np.argsort(gaps)[::-1][:5] if gaps[j] > 600]
    if longest:
        print("  the longest stretches without a block saved:")
        for j in sorted(longest):
            print(
                f"    {_format_time(mtimes[j])} to {_format_time(mtimes[j + 1])} "
                f"({_format_seconds(gaps[j])})"
            )


def _read_progress(checkpoint_dir):
    # the runs recorded in each rank's progress file
    progress_dir = os.path.join(checkpoint_dir, "progress")
    if not os.path.isdir(progress_dir):
        return {}
    runs = {}
    for name in sorted(os.listdir(progress_dir)):
        if not name.endswith(".log"):
            continue
        file_runs = runs[name.removesuffix(".log")] = []
        with open(os.path.join(progress_dir, name)) as f:
            for line in f:
                stamp, _, rest = line.rstrip("\n").partition(" ")
                event, _, rest = rest.partition(" ")
                try:
                    stamp = datetime.datetime.fromisoformat(stamp)
                except ValueError:
                    # e.g., a line cut short when the job was killed
                    continue
                if event == "begin":
                    info = dict(item.partition("=")[::2] for item in rest.split())
                    file_runs.append(
                        {
                            "begin": stamp,
                            "run": info.pop("run", None),
                            "info": " ".join(f"{k}={v}" for k, v in info.items()),
                            "started": {},
                            "done": [],
                        }
                    )
                if not file_runs:
                    continue
                run = file_runs[-1]
                run["last"] = stamp
                if event == "start":
                    i, _, about = rest.partition(" ")
                    run["started"][int(i)] = (stamp, about)
                elif event == "done":
                    i, seconds = rest.split()
                    _, about = run["started"].pop(int(i), (None, ""))
                    run["done"].append((float(seconds), int(i), about))
                elif event == "end":
                    run["end"] = rest
    return runs


def _print_progress(runs, saved, now):
    print()
    latest = max(
        (file_runs[-1] for file_runs in runs.values() if file_runs),
        key=lambda run: run["begin"],
        default=None,
    )
    if latest is None:
        print("no progress records (the export was started by an older version)")
        return
    print(
        f"progress records of the latest run ({latest['run']}), by rank; the "
        "ranks have one record file each, which every run appends to:"
    )
    durations = []
    for name, file_runs in runs.items():
        for run in file_runs:
            durations += run["done"]
        if not file_runs:
            continue
        run = file_runs[-1]
        if run["run"] != latest["run"]:
            print(
                f"  {name}: not in the latest run; in run {run['run']}, which "
                f"began {run['begin']:%Y-%m-%d %H:%M:%S}, it did "
                f"{len(run['done'])} blocks"
            )
            continue
        ended = run.get("end", "no end recorded: still running, or killed")
        print(
            f"  {name}: {len(file_runs)} run(s); the latest began "
            f"{run['begin']:%Y-%m-%d %H:%M:%S} ({run['info']}) and did "
            f"{len(run['done'])} blocks; {ended}; last record "
            f"{run['last']:%Y-%m-%d %H:%M:%S}, "
            f"{_format_seconds(now - run['last'].timestamp())} ago"
        )
        for i, (stamp, about) in run["started"].items():
            if i in saved:
                continue
            print(
                f"    unfinished: block {i} {about}, started {stamp:%Y-%m-%d %H:%M:%S}"
            )
    if durations:
        seconds = np.array([d[0] for d in durations])
        print(
            f"  {len(seconds)} blocks timed: {seconds.mean():.3g} s on average, "
            f"median {np.median(seconds):.3g} s, "
            f"{_format_seconds(seconds.sum())} in all; the slowest:"
        )
        for seconds, i, about in sorted(durations, reverse=True)[:5]:
            print(f"    block {i} {about}: {_format_seconds(seconds)}")


def _verify_blocks(store, blocks):
    keys = ("data", "mask", "left_edge", "right_edge", "dx", "min_val", "max_val")
    bad = []
    for i in blocks:
        try:
            block = store.load(i, keys)
            data, mask = np.array(block["data"].shape), np.array(block["mask"].shape)
            if (
                data.size != 3
                or mask.size != 3
                or not ((data == mask).all() or (data == mask + 1).all())
            ):
                raise ValueError(f"data of shape {data} with a mask of shape {mask}")
        except Exception as e:
            bad.append((i, e))
    print()
    print(f"read {len(blocks)} block files: {len(blocks) - len(bad)} are complete")
    for i, e in bad[:20]:
        print(f"  block {i}: {type(e).__name__}: {e}")


def main():
    if sys.argv[1:2] == ["--inspect"]:
        parser = argparse.ArgumentParser(
            prog=f"{os.path.basename(sys.argv[0])} --inspect",
            description="Report on an export's checkpoint directory; needs "
            "neither yt nor the dataset, and can run while the export does.",
        )
        parser.add_argument("checkpoint_dir", help="the export's checkpoint directory")
        parser.add_argument(
            "--verify", action="store_true", help="also read every block file"
        )
        args = parser.parse_args(sys.argv[2:])
        inspect_checkpoint(args.checkpoint_dir, verify=args.verify)
        return

    if sys.argv[1:2] == ["--assemble"]:
        parser = argparse.ArgumentParser(
            prog=f"{os.path.basename(sys.argv[0])} --assemble",
            description="Write the scene file of an export whose blocks were "
            "all saved to a checkpoint directory; needs neither yt nor the dataset.",
        )
        parser.add_argument("checkpoint_dir", help="the export's checkpoint directory")
        parser.add_argument("output", help="the scene file to write")
        parser.add_argument(
            "--log-interval",
            type=float,
            default=300,
            help="seconds between progress reports (0 for none)",
        )
        args = parser.parse_args(sys.argv[2:])
        assemble_scene(args.checkpoint_dir, args.output, args.log_interval)
        return

    import yt

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="the dataset to load with yt.load")
    parser.add_argument(
        "field", help="the field to render, e.g. density or gas,density"
    )
    parser.add_argument("output", help="the scene file to write")
    parser.add_argument("--no-ghost", action="store_true", help="skip ghost zones")
    parser.add_argument("--render-method", default=None, help="e.g. projection")
    parser.add_argument("--compress", action="store_true", help="compress the file")
    parser.add_argument(
        "--float64",
        action="store_true",
        help="store the geometry and texture data in float64 (larger file)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="save blocks here as they are generated, so that running the same "
        "command again resumes an interrupted export (or --assemble writes the "
        "scene file from it, once every block is saved)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="generate the blocks in parallel with MPI (run with mpirun)",
    )
    parser.add_argument(
        "--no-assemble",
        action="store_true",
        help="stop once every block is saved to the checkpoint directory, and "
        "leave the scene file to --assemble",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=300,
        help="seconds between progress reports (0 for none)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report what the export would do, reading the dataset's index "
        "but no field data (unless --sample is given)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="with --dry-run: the number of MPI ranks to plan for",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="with --dry-run: generate this many blocks, without saving them, "
        "to estimate how long the export would take",
    )
    args = parser.parse_args()

    from yt.funcs import is_root

    if args.parallel and not args.dry_run:
        yt.enable_parallelism()
    started = time.monotonic()
    ds = yt.load(args.dataset)
    n_grids = len(ds.index.grids)
    if is_root():
        _log(
            f"loaded {ds} and its index in "
            f"{_format_seconds(time.monotonic() - started)}: {n_grids} grids, "
            f"levels 0 to {ds.index.max_level}"
        )
    field = tuple(args.field.split(",")) if "," in args.field else args.field
    if args.dry_run:
        diagnose_export(
            ds.all_data(),
            field,
            no_ghost=args.no_ghost,
            float64=args.float64,
            checkpoint_dir=args.checkpoint_dir,
            n_ranks=args.ranks,
            n_sample=args.sample,
        )
        return
    export_block_scene(
        ds.all_data(),
        field,
        args.output,
        no_ghost=args.no_ghost,
        render_method=args.render_method,
        compress=args.compress,
        float64=args.float64,
        checkpoint_dir=args.checkpoint_dir,
        assemble=not args.no_assemble,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
