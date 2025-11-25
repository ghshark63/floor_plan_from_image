# Floor Plan Generator

This project generates a 2D floor plan from a video input or existing 3D reconstruction data. It utilizes 3D reconstruction, point cloud processing, and object detection to create a labeled floor plan.

## Prerequisites

### Python Dependencies

The project requires Python 3.11 or higher. Install the dependencies using Poetry or pip:

```bash
pip install open3d opencv-python ultralytics pycolmap trimesh pillow
```

### OpenMVS

This project relies on **OpenMVS** for dense reconstruction and texturing. You need to build OpenMVS or install it via `vcpkg`.

**Important:** The OpenMVS binaries must be placed in the following directory:
`src/thirdparty/openmvs/make`

Ensure the following executables are present in that folder:

- `InterfaceCOLMAP`
- `DensifyPointCloud`
- `ReconstructMesh`
- `RefineMesh`
- `TextureMesh`

## Usage

The main entry point for the pipeline is `src/main_pipeline.py`.

### Running from Video

To run the full pipeline starting from a video file:

```bash
python src/main_pipeline.py --reconstruct --input_video path/to/video.mp4 --output_dir output
```

### Running from Existing Reconstruction

If you already have a 3D reconstruction (point cloud, sparse model, and images), you can skip the reconstruction step:

```bash
python src/main_pipeline.py \
    --point_cloud output/final.ply \
    --sparse_dir output/sparse \
    --images_dir output/images \
    --texture_path output/textured0.png \
    --output_dir output
```

### Arguments

| Argument              | Description                                        | Default                     |
| --------------------- | -------------------------------------------------- | --------------------------- |
| `--reconstruct`       | Flag to run 3D reconstruction from video.          | `False`                     |
| `--input_video`       | Path to the input video file.                      | `input/video.mp4`           |
| `--output_dir`        | Directory where all output files will be saved.    | `output`                    |
| `--images_dir`        | Directory containing extracted images.             | `output_dir/images`         |
| `--sparse_dir`        | Directory containing COLMAP sparse reconstruction. | `output_dir/sparse`         |
| `--point_cloud`       | Path to the dense point cloud file (`.ply`).       | `output_dir/final.ply`      |
| `--texture_path`      | Path to the texture map image.                     | `output_dir/textured0.png`  |
| `--floor_plan_output` | Path for the final floor plan image.               | `output_dir/floor_plan.png` |

## Input & Output

### Input

- **Video:** A video file (e.g., `.mp4`) scanning the room.
- **(Alternative) Reconstruction Data:**
  - Dense Point Cloud (`.ply`)
  - Sparse Reconstruction (COLMAP format)
  - Extracted Images

### Output

- **Floor Plan:** A 2D image (`floor_plan.png`) showing the room layout and detected furniture.
- **Intermediate Files:**
  - Extracted frames from video
  - Sparse reconstruction data
  - Dense point cloud
  - Textured mesh
