import os
import sys
import argparse
import numpy as np
import open3d as o3d
import traceback

# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import FloorPlanGeneratorConfig
from main_pipeline import FloorPlanGenerator, load_cameras_from_colmap

def preview_mesh():
    parser = argparse.ArgumentParser(description="Preview mesh as it will appear in the final floor plan")
    
    # Path arguments (consistent with main_pipeline.py)
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--sparse_dir", type=str, help="Path to sparse reconstruction directory (default: output_dir/sparse)")
    parser.add_argument("--point_cloud", type=str, help="Path to point cloud file (default: output_dir/final.ply)")
    parser.add_argument("--texture_path", type=str, help="Path to texture file (default: output_dir/textured0.png)")
    parser.add_argument("--preview_output", type=str, help="Path to output preview image (default: output_dir/preview_floor_plan.png)")
    
    args = parser.parse_args()

    # Resolve paths
    output_dir = args.output_dir
    sparse_dir = args.sparse_dir if args.sparse_dir else os.path.join(output_dir, "sparse")
    point_cloud_path = args.point_cloud if args.point_cloud else os.path.join(output_dir, "final.ply")
    texture_path = args.texture_path if args.texture_path else os.path.join(output_dir, "textured0.png")
    preview_output = args.preview_output if args.preview_output else os.path.join(output_dir, "preview_floor_plan.png")
    
    # We assume the mesh to visualize is the same as the point cloud path (usually final.ply is a mesh file that can be read as point cloud too)
    # But FloorPlanVisualizer expects a mesh path.
    # In main_pipeline, config.floor_plan.mesh_path is used.
    # We should probably use point_cloud_path as mesh_path if it is a ply file.
    mesh_path = point_cloud_path

    sparse_model_path = os.path.join(sparse_dir, "0")
    
    print(f"Using Point Cloud/Mesh: {point_cloud_path}")
    print(f"Using Sparse Model: {sparse_model_path}")
    print(f"Using Texture: {texture_path}")
    
    if not os.path.exists(point_cloud_path):
        print(f"Error: Point cloud/mesh not found at {point_cloud_path}")
        return

    # Load cameras from COLMAP
    try:
        cameras = load_cameras_from_colmap(sparse_model_path)
    except Exception as e:
        print(f"Error loading cameras from {sparse_model_path}: {e}")
        return
    
    # Initialize Config
    # We need camera intrinsics and image size from cameras
    if not cameras:
        print("Error: No cameras found.")
        return

    config = FloorPlanGeneratorConfig.default(
        camera_intrinsics=cameras[0]['intrinsics'], 
        image_size=(cameras[0]['image_width'], cameras[0]['image_height']),
        texture_path=texture_path
    )
    
    # Override mesh path in config to point to our target mesh
    config.floor_plan.mesh_path = mesh_path

    # Initialize Generator (without YOLO)
    generator = FloorPlanGenerator(config, yolo_detector=None)

    try:
        # 1. Load Point Cloud for Alignment Calculation
        print("Loading point cloud for alignment calculation...")
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
        
        # 2. Perform Alignment
        # Align point cloud to canonical orientation based on camera orientations
        point_cloud, R_cam = generator._align_reconstruction(point_cloud, cameras)
        
        # Refine alignment using floor plane
        point_cloud, R_floor = generator._align_to_floor(point_cloud, cameras)
        
        # Refine alignment using wall planes (align to axes)
        point_cloud, R_walls = generator._align_to_walls(point_cloud, cameras)
        
        alignment_rotation = R_walls @ R_floor @ R_cam
        print(f"Total alignment rotation:\n{alignment_rotation}")
        
        # 3. Preprocess and Segment (to get wall planes for removal)
        print("Segmenting structure to detect walls...")
        processed_pcd = generator.point_cloud_processor.preprocess(point_cloud)
        segmentation_results = generator.segmenter.segment(processed_pcd)
        
        # 4. Generate Preview
        print(f"Generating preview to {preview_output}...")
        
        # We pass empty list for clusters to skip furniture
        generator.floor_plan_visualizer.generate_floor_plan(
            clusters=[], 
            mesh_path=config.floor_plan.mesh_path,
            texture_path=config.floor_plan.texture_path,
            output_path=preview_output,
            alignment_matrix=alignment_rotation,
            wall_planes=segmentation_results.wall_planes
        )
        
        print("Preview generation completed successfully!")

    except Exception as e:
        traceback.print_exc()
        print(f"Preview generation failed: {e}")

if __name__ == "__main__":
    preview_mesh()
