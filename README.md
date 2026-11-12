# yt_idv_export

Export a [yt](https://yt-project.org) data source as a scene file that
[yt_idv](https://github.com/yt-project/yt_idv) can open, on a machine that has
neither yt_idv nor OpenGL (e.g. a cluster's compute nodes).

It is a single module, `yt_idv_export.py`, that needs only numpy and yt
(plus mpi4py for parallel exports). It can be installed, or copied and run on
its own.

Install it as a command-line tool with [uv](https://docs.astral.sh/uv/)
(or pipx), or into an environment with pip:

```bash
uv tool install /path/to/yt_idv_export          # or a git URL
uv tool install "/path/to/yt_idv_export[mpi]"   # with mpi4py, for --parallel
pip install .
```

```bash
yt-idv-export DATASET FIELD scene.zip [--no-ghost] [--periodic] [--float64] [--float64-textures]
```

`--periodic` treats the dataset as periodic along every axis, whatever its own
periodicity, so that the ghost zones at the domain's boundaries are filled from
the opposite side (from python, call `ds.force_periodicity()` before exporting).

`--float64` stores the blocks' edges and cell widths in float64, which keeps
deep zooms exact and adds little to the file. `--float64-textures` also stores
the texture data in float64, which doubles the textures' size; yt_idv still
sends float32 to the GPU.

The blocks are the leaves of yt's kd tree: they cover the data source once,
each part with the finest data available there. When yt inserts a finer grid
into a coarser one, the kd tree node it splits keeps referring to the coarser
grid. With `--coarse-nodes` the scene file also stores the coarser data at the
topmost such node of each grid, covering that node's whole subtree. This is
the start of a level-of-detail hierarchy: a renderer could draw a coarse block
in place of the finer blocks below it. The extra attributes are
`_coarse_block_data`, `_coarse_block_bitmaps`, `_coarse_edges` (unitary
float64 left edges, right edges and cell widths) and `_kd_coarse_block` (the
coarse block at each `_kd_tree` entry, or -1). The leaves are written exactly
as without the option, so yt_idv renders the scene as usual and ignores the
coarse blocks.

```python
from yt_idv_export import export_block_scene
export_block_scene(ds.all_data(), "density", "scene.zip")
```

Then, wherever yt_idv is available:

```python
import yt_idv
rc = yt_idv.render_context(height=800, width=800)
rc.scene = yt_idv.scene_graph.SceneGraph.load("scene.zip")
rc.run()
```

Checkpointing and resuming (`--checkpoint-dir`), MPI (`--parallel`),
assembling from a checkpoint without yt (`--assemble`), `--dry-run` and
`--inspect` are described in the module docstring and `yt-idv-export --help`.

## Compatibility with yt_idv

The scene file format (`FORMAT_VERSION`) and the yt_idv class paths written
into it are fixed in this module. Exports with `--float64-textures` need a
yt_idv that can load float64 textures.

## License

BSD 3-Clause; see `LICENSE`.
