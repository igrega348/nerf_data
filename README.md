# nerf_data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/igrega348/nerf_data/HEAD?labpath=examples%2Fgetting_started.ipynb)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Data preparation and utility toolkit for X-ray NeRF (Neural Radiance Field) reconstruction. Converts raw X-ray CT hardware output into training-ready datasets for [nerfstudio-xray](https://github.com/igrega348/nerfstudio-xray), and provides volumetric format converters, visualization tools, and reference synthetic datasets.

## Contents

```
nerf_data/
├── scripts/                    # CLI utilities (see Scripts reference below)
│   ├── compute_transforms.py   # X-ray hardware metadata → transforms.json
│   ├── combine_transforms.py   # Merge per-timestep transforms into one file
│   ├── raw_to_npy.py           # Raw binary volume → .npy / .npz
│   ├── npy_to_raw.py           # .npy / .npz → raw binary volume
│   ├── tiff_to_png.py          # TIFF projections → PNG with thresholding
│   ├── color_convert.py        # Batch convert images to grayscale
│   ├── resize_for_eval.py      # Downscale images for evaluation split
│   ├── read_flat_field.py      # Interactive flat-field ROI selection
│   ├── show_slices.py          # Interactive 3D slice viewer (single window)
│   ├── show_slices3.py         # 3D slice viewer (three separate windows)
│   ├── show_deformation.py     # Deformation / velocity field visualizer
│   ├── show_hist.py            # Volumetric data histogram
│   ├── nerf_xray_importer.py   # Blender add-on: import object YAML/JSON
│   ├── combine_transforms.ipynb  # Notebook: combine time-series transforms
│   ├── scale_lattice_yaml.ipynb  # Notebook: scale lattice YAML geometry
│   ├── kelvin.yaml             # Sample Kelvin-lattice unit cell definition
│   └── lattice.yaml            # Sample lattice unit cell definition
├── synthetic/                  # Pre-computed synthetic datasets
│   ├── balls/                  # Six spheres; 281-frame transforms.json + YAML
│   ├── balls_grey/             # Greyscale variant
│   ├── cube.tar.gz             # Synthetic cube
│   └── pillars.tar.gz          # Synthetic cylindrical pillars
├── experimental/               # Real X-ray CT datasets
│   ├── balls/                  # Nikon CT scan of steel balls
│   ├── balls7/                 # Alternative acquisition of same sample
│   ├── balls7_th/              # Thresholded variant
│   ├── cube/                   # Cube sample
│   └── pillars/                # Cylindrical pillars sample
├── examples/                   # Runnable notebooks and example scripts
│   └── getting_started.ipynb   # Interactive tour of the data pipeline
├── binder/                     # Binder environment config
│   └── requirements.txt
└── requirements.txt            # Full dependency list
```

## Quick start

```bash
git clone https://github.com/igrega348/nerf_data.git
cd nerf_data
pip install -r requirements.txt
```

All scripts use [tyro](https://github.com/brentyi/tyro) for structured CLIs — run any script with `--help` for the full argument list:

```bash
python scripts/compute_transforms.py --help
python scripts/raw_to_npy.py --help
```

## Scripts reference

### Data ingestion from real X-ray CT hardware

#### `compute_transforms.py` — hardware metadata → `transforms.json`

Reads the scanner configuration (`.xtekct`) and rotation-angle log (`.ang` or `_ctdata`) produced by Nikon/Metris µCT systems and writes a NeRF-compatible `transforms.json`.

```bash
python scripts/compute_transforms.py \
  path/to/scan_dir \                   # positional: directory with .xtekct and .ang/.ctdata
  --output-fname transforms.json \     # output filename (default: transforms.json)
  --images-folder images \             # subfolder containing projection PNGs (default: images)
  --deblurring Gauss \                 # motion-blur correction: Gauss | uniform | omit to disable
  --deblurring-points 7 \             # quadrature points for deblurring (default: 7)
  --time 0.0 \                         # optional: assign a fixed time value to all frames
  --flat-field 1.0                     # optional: flat-field correction value written to JSON
```

The camera distance `R` and all intrinsics (focal length, FoV, principal point) are computed automatically from the `.xtekct` file using `SrcToObject`, `SrcToDetector`, and `DetectorPixelSize`. No manual geometry parameters are needed.

#### `combine_transforms.py` — merge per-timestep files into one

After generating one `transforms_NN.json` per load step, combine them into a single file for time-resolved NeRF training:

```bash
python scripts/combine_transforms.py \
  --folder path/to/timestep_dir \             # contains transforms_00.json, transforms_01.json, …
  --timestamp-func "lambda x: x/20.0"        # optional: map integer step → normalised time [0,1]
```

Output: `folder/transforms.json` with a `time` field on every frame.

> **Note:** frames whose image files do not exist on disk are silently dropped (the script prints a warning per dropped frame). If all frames disappear, check that `--folder` points to the directory that contains the `images_NN/` subfolders.

### Format conversions

#### `raw_to_npy.py` — raw binary volume → numpy

```bash
python scripts/raw_to_npy.py \
  --input volume.raw \
  --resolution 512 512 512 \      # Nx Ny Nz of the input file (x, y, z order)
  --dtype UINT16 \                # input element dtype (default: UINT8)
  --output volume.npz \           # default: same path with .npz extension
  --out-resolution 128 128 128 \  # optional: resample to this resolution
  --thresholds 0.02 0.98          # optional: clip to [2%, 98%] of the value range (not percentiles)
```

The raw file is read as flat binary (`dtype`, row-major) and reshaped to `(Nx, Ny, Nz)` with axes ordered `(x, y, z)`. Output `.npz` stores the array under the key `vol`.

`--thresholds low high` clips at `min + low*(max-min)` and `min + high*(max-min)` — fractions of the observed value range, not true distribution percentiles.

#### `npy_to_raw.py` — numpy → raw binary volume

```bash
python scripts/npy_to_raw.py \
  --input volume.npz \
  --output volume_out.raw \
  --out-dtype UINT8 \            # optional: convert dtype on write
  --out-resolution 256 256 256   # optional: resample before writing
```

Axes are swapped to `(z, y, x)` on write to match the row-major layout expected by the Go X-ray renderer.

### Image processing

| Script | Purpose | Key flags |
|---|---|---|
| `tiff_to_png.py` | Convert TIFF projections to PNG | `--threshold`, `--colormap`, `--greyscale_func` |
| `color_convert.py` | Batch convert images to grayscale | `--pattern`, `--out_folder` |
| `resize_for_eval.py` | Downscale images for eval split | `--factor` |
| `read_flat_field.py` | Interactive ROI → flat-field value | `--image` |

### Visualization

| Script | Purpose | Key controls |
|---|---|---|
| `show_slices.py` | Single-window orthogonal slice viewer (x/y/z) | Trackbars for slice position and threshold percentiles |
| `show_slices3.py` | Three-window slice viewer | Same, simpler interface |
| `show_deformation.py` | Deformation / velocity field (RdBu colourmap) | `--component` to select vector component |
| `show_hist.py` | Histogram of volumetric data | `--log`, `--downsample` |

All viewers accept `.npy`, `.npz`, or `.raw` (requires `--resolution` for raw files).

### Blender integration

`nerf_xray_importer.py` is a Blender add-on that imports geometry from YAML/JSON scene definitions directly into a Blender scene. Supported primitives: sphere, cube, cylinder, box, parallelepiped, object_collection (recursive). Install via *Edit → Preferences → Add-ons → Install from file*.

## Data formats

### `transforms.json` — camera poses (NeRF format)

```json
{
  "camera_angle_x": 0.698,   // horizontal FoV in radians
  "fl_x": 686.87,            // focal length x (pixels)
  "fl_y": 686.87,            // focal length y (pixels)
  "w": 500,  "h": 500,       // image dimensions
  "cx": 250, "cy": 250,      // principal point
  "frames": [
    {
      "file_path": "images/train_000.png",
      "time": 0.0,                    // normalised time in [0, 1] (4D datasets only)
      "transform_matrix": [           // 4×4 camera-to-world matrix (OpenCV convention)
        [-1, 0, 0, 0],
        [ 0, 0, 1, 4.0],             // column 3 = camera origin (x, y, z)
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]
      ]
    }
  ]
}
```

**transform_matrix layout:**
- `M[:3, :3]` — rotation, camera-to-world (OpenCV convention; camera looks along local +Z)
- `M[:3, 3]` — camera origin in world coordinates
- `time` — present only in multi-timestep files; `0.0` = undeformed state, `1.0` = fully deformed

**Train/eval split** is encoded in the filename: `train_*` frames go to the training set, `eval_*` frames to the evaluation set (parsed by the `filename+modulo` eval mode in nerfstudio-xray).

**Multi-timestep files** (e.g. `transforms_00_to_20.json`) contain frames from all timesteps interleaved, each with its own `time` value.

### Object definition YAML — scene geometry

Used for forward rendering (X-ray projection generation) and volumetric supervision during training.

```yaml
type: object_collection
objects:
  - type: sphere
    center: [0.0, 0.0, 0.0]
    radius: 0.15
    rho: 1.0          # linear X-ray attenuation coefficient (Beer-Lambert)
  - type: cylinder
    p0: [0.0, 0.0, -0.5]
    p1: [0.0, 0.0,  0.5]
    radius: 0.05
    rho: 1.5
  - type: box
    center: [0.5, 0.0, 0.0]
    sides: [0.2, 0.3, 0.4]  # total side lengths (full extents)
    rho: 0.8
```

Supported primitives: `sphere`, `cylinder`, `box`, `parallelepiped`, `object_collection` (recursive). Tessellated lattice structures use `type: tessellated_obj_coll` with a `uc: unit_cell` sub-object.

### Volumetric arrays — raw / npy / npz

| Format | Shape | Axis order | Notes |
|---|---|---|---|
| `.raw` (input from CT scanner) | flat uint8/16 | `(Nz, Ny, Nx)` = `(z, y, x)` | Written by Nikon software |
| `.npy` / `.npz` (numpy) | `(Nx, Ny, Nz)` | `(x, y, z)` | `raw_to_npy.py` applies axis swap on load |
| `.raw` (output to renderer) | flat uint8 | `(Nz, Ny, Nx)` = `(z, y, x)` | `npy_to_raw.py` swaps axes back on write |

The `.npz` key is always `vol`. The Go X-ray renderer expects uint8 raw files in `(z, y, x)` order, so **always use `npy_to_raw.py` rather than `np.tofile()` directly**.

## Datasets

### Synthetic

Pre-computed projections rendered with the Go X-ray renderer.

| Dataset | Description | Frames |
|---|---|---|
| `synthetic/balls` | 6 spheres, radii 0.15–0.20, random positions | 281 (train + eval) |
| `synthetic/balls_grey` | Same geometry, greyscale images | — |
| `synthetic/cube.tar.gz` | Single cube | — |
| `synthetic/pillars.tar.gz` | Cylindrical pillars array | — |

Unpack tarballs with `tar -xzf *.tar.gz -C .`.

### Experimental

Real µCT data acquired on a Nikon XT H 225 system.

| Dataset | Description | Notes |
|---|---|---|
| `experimental/balls` | Steel balls phantom (257 projections) | Includes `.xtekct` + `.ang` hardware files |
| `experimental/balls7` | Same phantom, different acquisition | — |
| `experimental/balls7_th` | Thresholded version | — |
| `experimental/cube` | Solid cube | — |
| `experimental/pillars` | Cylindrical pillars | — |

The `experimental/balls/` directory also contains the raw hardware files (`*.xtekct`, `*.ang`) that can be used as a worked example for `compute_transforms.py`.

## Examples

Open the interactive notebook in your browser with no local setup:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/igrega348/nerf_data/HEAD?labpath=examples%2Fgetting_started.ipynb)

Or run locally:

```bash
pip install jupyter matplotlib numpy scipy pyyaml
jupyter notebook examples/getting_started.ipynb
```

The notebook covers:
- Loading and inspecting `transforms.json`
- Plotting camera positions and trajectories in 3D
- Parsing and visualising an object-collection YAML
- Understanding the raw ↔ npy axis-order convention
- Walking through how `compute_transforms.py` builds a camera matrix from hardware metadata

## Integration with nerfstudio-xray

This repo sits at the **data preparation** stage of the full pipeline:

```
Raw CT scan (.xtekct + .ang + TIFFs)
        │
        ▼  compute_transforms.py + tiff_to_png.py
transforms.json + PNG projections
        │
        ▼  (train NeRF)  nerfstudio-xray
Reconstructed 3D / 4D volume
        │
        ▼  show_slices.py / show_deformation.py
Visualisation
```

For 4D (time-resolved) datasets, multiple per-timestep `transforms_NN.json` files are first combined with `combine_transforms.py` before training.

See [nerfstudio-xray](https://github.com/igrega348/nerfstudio-xray) for the full training pipeline.

## License

GPL v3 — see [LICENSE](LICENSE).
