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
yt-idv-export DATASET FIELD scene.zip [--no-ghost] [--float64] [--float64-textures]
```

`--float64` stores the blocks' edges and cell widths in float64, which keeps
deep zooms exact and adds little to the file. `--float64-textures` also stores
the texture data in float64, which doubles the textures' size; yt_idv still
sends float32 to the GPU.

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
