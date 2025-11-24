import os
import subprocess
import sys
from pathlib import Path
import shutil

# Add current directory to path to allow imports if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from vid2img import video_to_images
except ImportError:
    # Fallback if running from root
    try:
        from src.vid2img import video_to_images
    except ImportError:
        # If we are in src but vid2img is not found (should not happen if in same dir)
        pass

import pycolmap

class ReconstructionPipeline:
    def __init__(self, 
                 root_dir: str = ".",
                 input_video: str = None,
                 output_dir: str = None,
                 images_dir: str = None,
                 sparse_dir: str = None,
                 texture_file_name: str = "textured0.png"):
        self.root_dir = Path(root_dir).resolve()
        
        # Default paths if not provided
        self.input_video = Path(input_video).resolve() if input_video else self.root_dir / "input" / "video.mp4"
        self.output_dir = Path(output_dir).resolve() if output_dir else self.root_dir / "output"
        self.images_dir = Path(images_dir).resolve() if images_dir else self.output_dir / "images"
        self.sparse_dir = Path(sparse_dir).resolve() if sparse_dir else self.output_dir / "sparse"
        self.database_path = self.output_dir / "database.db"
        self.texture_file_name = texture_file_name
        
        # Binaries path
        self.mvs_bin_dir = self.root_dir / "src" / "thirdparty" / "openmvs" / "make"
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Convert video to images
        print("Step 1: Converting video to images...")
        if self.input_video.exists():
             # Use 5 FPS as default
             video_to_images(str(self.input_video), str(self.images_dir), 5.0)
        else:
            print(f"Warning: {self.input_video} not found. Checking if images exist...")
            if not any(self.images_dir.iterdir()):
                print("Error: No input video and no images found.")
                return

        # 2. COLMAP Sparse Reconstruction
        print("Step 2: Running COLMAP sparse reconstruction...")
        if self.database_path.exists():
            try:
                os.remove(self.database_path)
            except OSError:
                pass
            
        # Pycolmap pipeline
        print("Extracting features...")
        pycolmap.extract_features(self.database_path, self.images_dir)
        
        print("Matching features...")
        pycolmap.match_exhaustive(self.database_path)
        
        print("Running mapper...")
        # incremental_mapping returns a dict of reconstructions
        # and saves them to output_path if provided
        pycolmap.incremental_mapping(self.database_path, self.images_dir, self.sparse_dir)
        
        # Check if reconstruction was successful
        # COLMAP usually creates '0', '1' folders in sparse_dir
        reconstruction_path = self.sparse_dir / "0"
        if not reconstruction_path.exists():
            print("Error: COLMAP reconstruction failed or produced no model.")
            return

        # 3. OpenMVS Pipeline
        print("Step 3: Running OpenMVS pipeline...")
        
        # Define binary paths (assuming Windows .exe)
        interface_colmap = self.mvs_bin_dir / "InterfaceCOLMAP.exe"
        densify_point_cloud = self.mvs_bin_dir / "DensifyPointCloud.exe"
        reconstruct_mesh = self.mvs_bin_dir / "ReconstructMesh.exe"
        refine_mesh = self.mvs_bin_dir / "RefineMesh.exe"
        texture_mesh = self.mvs_bin_dir / "TextureMesh.exe"
        
        # Helper to run command
        def run_cmd(cmd_list):
            print(f"Running: {' '.join(str(x) for x in cmd_list)}")
            subprocess.check_call([str(x) for x in cmd_list])

        # InterfaceCOLMAP
        scene_mvs = self.output_dir / "scene.mvs"
        run_cmd([
            interface_colmap,
            "-i", reconstruction_path,
            "-o", scene_mvs,
            "--image-folder", self.images_dir
        ])
        
        # DensifyPointCloud
        scene_dense = self.output_dir / "scene_dense.mvs"
        run_cmd([
            densify_point_cloud,
            scene_mvs,
            "-o", scene_dense
        ])
        
        # ReconstructMesh
        scene_mesh = self.output_dir / "scene_dense_mesh.mvs"
        run_cmd([
            reconstruct_mesh,
            scene_dense,
            "-o", scene_mesh
        ])
        
        # RefineMesh
        scene_mesh_refined = self.output_dir / "scene_dense_mesh_refined.mvs"
        run_cmd([
            refine_mesh,
            scene_mesh,
            "-o", scene_mesh_refined
        ])
        
        # TextureMesh
        final_ply = self.output_dir / "final.ply"
        run_cmd([
            texture_mesh,
            scene_mesh_refined,
            "-o", final_ply,
            "--export-type", "ply"
        ])
        
        # Check for texture file and rename if necessary
        # OpenMVS usually names it based on output filename, e.g., final.png
        # User wants 'textured0.png' (or whatever is passed)
        # If final.ply is generated, check for final.png
        expected_texture = self.output_dir / "final.png"
        target_texture = self.output_dir / self.texture_file_name
        
        if expected_texture.exists():
            if target_texture.exists() and target_texture != expected_texture:
                os.remove(target_texture)
            
            if target_texture != expected_texture:
                os.rename(expected_texture, target_texture)
                print(f"Renamed texture to {target_texture}")
        else:
            print(f"Warning: Expected texture file {expected_texture} not found.")

if __name__ == "__main__":
    pipeline = ReconstructionPipeline()
    pipeline.run()
