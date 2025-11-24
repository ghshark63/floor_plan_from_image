import numpy as np
from typing import List, Dict, Any
import open3d as o3d
import glob
import cv2
import os
import argparse
from ultralytics import YOLO
import pycolmap

from config import FloorPlanGeneratorConfig
from point_cloud_processor import PointCloudProcessor
from furniture_clusterer import FurnitureClusterer
from label_projector import LabelProjector
from segmentation import StructuralSegmenter
from visualizer import FloorPlanVisualizer
from yolo_wrapper import YOLODetector
from reconstruction import ReconstructionPipeline
import traceback


class FloorPlanGenerator:
    def __init__(self, config: FloorPlanGeneratorConfig, yolo_detector: YOLODetector = None):
        self.config = config
        self.yolo_detector = yolo_detector
        self.point_cloud_processor = PointCloudProcessor(config.point_cloud)
        self.segmenter = StructuralSegmenter(config.segmentation)
        self.furniture_clusterer = FurnitureClusterer(config.clustering)
        self.label_projector = LabelProjector(config.labeling)
        self.floor_plan_visualizer = FloorPlanVisualizer(config.floor_plan)


    def run_pipeline(self,
                     point_cloud_path: str,
                     images: List[np.ndarray],
                     cameras: List[Dict[str, Any]],
                     output_path: str = "floor_plan.png"):
        # Get point cloud
        print("Loading point cloud...")
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        print(f"Loaded point cloud from {point_cloud_path} with {len(point_cloud.points)} points.")
        
        # Align point cloud to canonical orientation based on camera orientations
        point_cloud, R_cam = self._align_reconstruction(point_cloud, cameras)
        
        # Refine alignment using floor plane
        point_cloud, R_floor = self._align_to_floor(point_cloud, cameras)
        
        # Refine alignment using wall planes (align to axes)
        point_cloud, R_walls = self._align_to_walls(point_cloud, cameras)
        
        alignment_rotation = R_walls @ R_floor @ R_cam
        print(f"Total alignment rotation:\n{alignment_rotation}")
        
        processed_pcd = self.point_cloud_processor.preprocess(point_cloud)

        # Detect furniture in 2d
        detections = self.yolo_detector.detect_furniture(images)

        segmentation_results = self.segmenter.segment(processed_pcd)
        furniture_points = segmentation_results.furniture_points

        # New pipeline: Label points first, then cluster
        furniture_points_np = np.asarray(furniture_points.points)
        
        point_labels, _ = self.label_projector.project_labels_to_points(
            furniture_points_np, detections, cameras
        )
        
        # Filter out unknown points
        valid_indices = [i for i, label in enumerate(point_labels) if label != 'unknown']
        
        if not valid_indices:
            print("No furniture points labeled. Skipping clustering.")
            labeled_clusters = []
        else:
            cleaned_points = furniture_points_np[valid_indices]
            cleaned_labels = [point_labels[i] for i in valid_indices]
            
            print(f"Kept {len(cleaned_points)} points after filtering unknown labels")
            
            labeled_clusters = self.furniture_clusterer.cluster_labeled_points(
                cleaned_points, cleaned_labels
            )

        self.floor_plan_visualizer.generate_floor_plan(
            labeled_clusters,
            mesh_path=self.config.floor_plan.mesh_path,
            texture_path=self.config.floor_plan.texture_path,
            output_path=output_path,
            alignment_matrix=alignment_rotation,
            wall_planes=segmentation_results.wall_planes
        )

    def _align_reconstruction(self, point_cloud: o3d.geometry.PointCloud, 
                             cameras: List[Dict[str, Any]]) -> tuple:
        """
        Align reconstruction to canonical orientation using camera viewing directions.
        Assumes cameras look horizontally (at walls), not at floor/ceiling.
        """
        print("Aligning reconstruction to canonical orientation...")
        
        # Extract camera viewing directions (negative Z-axis in camera frame)
        # In camera frame, looking direction is typically -Z axis
        viewing_dirs = []
        up_dirs = []
        
        for cam in cameras:
            R = cam['rotation']
            # Camera coordinate system: X-right, Y-down, Z-forward
            # World viewing direction is R.T @ [0, 0, 1] (Z-axis of camera in world coords)
            viewing_dir = R.T @ np.array([0, 0, 1])
            up_dir = R.T @ np.array([0, -1, 0])  # Y-axis (inverted because camera Y is down)
            
            viewing_dirs.append(viewing_dir)
            up_dirs.append(up_dir)
        
        viewing_dirs = np.array(viewing_dirs)
        up_dirs = np.array(up_dirs)
        
        # Average up direction across all cameras
        mean_up = np.mean(up_dirs, axis=0)
        mean_up = mean_up / np.linalg.norm(mean_up)
        
        print(f"Current 'up' direction in world coords: {mean_up}")
        
        # Target up direction should be [0, 1, 0] (Y-axis up)
        target_up = np.array([0, 1, 0])
        
        # Compute rotation to align mean_up with target_up
        # Using Rodrigues' rotation formula
        v = np.cross(mean_up, target_up)
        s = np.linalg.norm(v)
        c = np.dot(mean_up, target_up)
        
        if s < 1e-6:  # Already aligned
            print("Reconstruction already aligned")
            return point_cloud, np.eye(3)
        
        # Skew-symmetric cross-product matrix
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        # Rotation matrix
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        
        print(f"Applying alignment rotation...")
        print(f"New 'up' direction: {R_align @ mean_up}")
        
        # Apply rotation to point cloud
        point_cloud.rotate(R_align, center=(0, 0, 0))
        
        # Update camera rotations to match
        for cam in cameras:
            cam['rotation'] = cam['rotation'] @ R_align.T
        
        return point_cloud, R_align

    def _align_to_floor(self, point_cloud: o3d.geometry.PointCloud, 
                       cameras: List[Dict[str, Any]]) -> tuple:
        """
        Refine alignment by detecting the floor plane and aligning its normal to [0, 1, 0].
        """
        print("Refining alignment using floor plane detection...")
        
        # Detect floor plane
        # We use the segmenter's detect_floor method directly
        try:
            _, floor_model = self.segmenter.detect_floor(point_cloud)
        except Exception as e:
            print(f"Floor detection failed during alignment: {e}")
            return point_cloud, np.eye(3)
            
        if floor_model is None:
            print("No valid floor plane found for alignment refinement.")
            return point_cloud, np.eye(3)
            
        # Extract normal
        a, b, c, d = floor_model
        floor_normal = np.array([a, b, c])
        floor_normal = floor_normal / np.linalg.norm(floor_normal)
        
        print(f"Detected floor normal: {floor_normal}")
        
        # Target up direction (Y-axis)
        target_up = np.array([0, 1, 0])
        
        # Check if normal points up or down. We want it to point UP (Y+)
        if np.dot(floor_normal, target_up) < 0:
            print("Flipping floor normal to match general up direction")
            floor_normal = -floor_normal
            
        # Compute rotation to align floor_normal with target_up
        v = np.cross(floor_normal, target_up)
        s = np.linalg.norm(v)
        c = np.dot(floor_normal, target_up)
        
        if s < 1e-6:
            print("Floor already aligned")
            return point_cloud, np.eye(3)
            
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R_refine = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        
        print(f"Applying refinement rotation...")
        print(f"New floor normal: {R_refine @ floor_normal}")
        
        # Apply rotation
        point_cloud.rotate(R_refine, center=(0, 0, 0))
        
        # Update cameras
        for cam in cameras:
            cam['rotation'] = cam['rotation'] @ R_refine.T
            
        return point_cloud, R_refine

    def _align_to_walls(self, point_cloud: o3d.geometry.PointCloud, 
                       cameras: List[Dict[str, Any]]) -> tuple:
        """
        Refine alignment by detecting walls and aligning them to X/Z axes.
        Assumes floor is already aligned to XZ plane (Y-up).
        """
        print("Refining alignment using wall plane detection...")
        
        try:
            # Re-detect structure in the currently aligned frame
            floor_indices, _ = self.segmenter.detect_floor(point_cloud)
            ceil_indices = self.segmenter.detect_ceil(point_cloud, floor_indices)
            _, wall_normals = self.segmenter.detect_walls(point_cloud, floor_indices, ceil_indices)
        except Exception as e:
            print(f"Wall detection failed during alignment: {e}")
            return point_cloud, np.eye(3)
            
        if not wall_normals:
            print("No walls found for alignment refinement.")
            return point_cloud, np.eye(3)
            
        # Project normals to XZ plane
        projected_normals = []
        for n in wall_normals:
            # Ignore Y component
            n_xz = np.array([n[0], n[2]])
            norm = np.linalg.norm(n_xz)
            if norm > 0.1: 
                projected_normals.append(n_xz / norm)
        
        if not projected_normals:
            return point_cloud, np.eye(3)
            
        projected_normals = np.array(projected_normals)
        
        # Calculate angles in degrees [0, 360)
        angles = np.degrees(np.arctan2(projected_normals[:, 1], projected_normals[:, 0]))
        angles = np.mod(angles, 360)
        
        # Map to [0, 90) to find dominant grid orientation
        # We assume walls are mostly orthogonal
        angles_mod = np.mod(angles, 90)
        
        # Use histogram to find peak
        hist, bin_edges = np.histogram(angles_mod, bins=90, range=(0, 90))
        best_bin = np.argmax(hist)
        best_angle_deg = (bin_edges[best_bin] + bin_edges[best_bin+1]) / 2
        
        print(f"Dominant wall orientation (mod 90): {best_angle_deg:.2f} degrees")
        
        # Rotate to align this peak to 0 degrees (X-axis)
        # We rotate around Y axis
        rotation_angle_rad = np.radians(-best_angle_deg)
        
        c = np.cos(rotation_angle_rad)
        s = np.sin(rotation_angle_rad)
        
        R_wall = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])
        
        print(f"Aligning walls by rotating {np.degrees(rotation_angle_rad):.2f} degrees around Y")
        
        point_cloud.rotate(R_wall, center=(0, 0, 0))
        
        for cam in cameras:
            cam['rotation'] = cam['rotation'] @ R_wall.T
            
        return point_cloud, R_wall


def load_cameras_from_colmap(sparse_path: str) -> List[Dict[str, Any]]:
    """
    Load camera parameters from COLMAP sparse reconstruction.
    
    Args:
        sparse_path: Path to COLMAP sparse directory (e.g., 'output/sparse/0')
    
    Returns:
        List of camera dictionaries with intrinsics, rotation, translation, width, height
    """
    reconstruction = pycolmap.Reconstruction(sparse_path)
    
    cameras = []
    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        
        # Get camera intrinsics
        fx = camera.focal_length_x
        fy = camera.focal_length_y
        cx = camera.principal_point_x
        cy = camera.principal_point_y
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Get camera pose (rotation and translation)
        # COLMAP stores world-to-camera transformation
        pose = image.cam_from_world()
        rotation = pose.rotation.matrix()
        translation = pose.translation
        
        cameras.append({
            'intrinsics': intrinsics,
            'rotation': rotation,
            'translation': translation,
            'image_width': camera.width,
            'image_height': camera.height,
            'image_name': image.name
        })
    
    print(f"Loaded {len(cameras)} cameras from COLMAP reconstruction")
    return cameras


def run():
    parser = argparse.ArgumentParser(description="Floor Plan Generator Pipeline")
    parser.add_argument("--reconstruct", action="store_true", help="Run 3D reconstruction from video")
    
    # Path arguments
    parser.add_argument("--input_video", type=str, default="input/video.mp4", help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--images_dir", type=str, help="Path to images directory (default: output_dir/images)")
    parser.add_argument("--sparse_dir", type=str, help="Path to sparse reconstruction directory (default: output_dir/sparse)")
    parser.add_argument("--point_cloud", type=str, help="Path to point cloud file (default: output_dir/final.ply)")
    parser.add_argument("--texture_path", type=str, help="Path to texture file (default: output_dir/textures.png)")
    parser.add_argument("--floor_plan_output", type=str, help="Path to output floor plan image (default: output_dir/floor_plan.png)")
    
    args = parser.parse_args()

    # Resolve paths
    input_video = args.input_video
    output_dir = args.output_dir
    
    images_dir = args.images_dir if args.images_dir else os.path.join(output_dir, "images")
    sparse_dir = args.sparse_dir if args.sparse_dir else os.path.join(output_dir, "sparse")
    point_cloud_path = args.point_cloud if args.point_cloud else os.path.join(output_dir, "point_cloud.ply")
    texture_path = args.texture_path if args.texture_path else os.path.join(output_dir, "textures.png")
    floor_plan_output = args.floor_plan_output if args.floor_plan_output else os.path.join(output_dir, "floor_plan.png")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if args.reconstruct:
        print("Starting 3D Reconstruction...")
        pipeline = ReconstructionPipeline(
            root_dir=".",
            input_video=input_video,
            output_dir=output_dir,
            images_dir=images_dir,
            sparse_dir=sparse_dir,
            texture_file_name=os.path.basename(texture_path)
        )
        pipeline.run()
        
        # After reconstruction, we expect the sparse model to be in sparse_dir/0
        # But ReconstructionPipeline puts it in sparse_dir/0
        sparse_model_path = os.path.join(sparse_dir, "0")
    else:
        # If not reconstructing, we assume the user provided paths to existing data
        # If they didn't provide specific paths, we use the defaults (which point to output/)
        # BUT, for backward compatibility or ease of use with test data, we might want to check if they exist?
        # The user asked to "make them consistent across the pipeline".
        # So we stick to the arguments.
        sparse_model_path = os.path.join(sparse_dir, "0")
        
        # If the user wants to use test data, they should provide:
        # --point_cloud output/point_cloud.ply --sparse_dir output/sparse --texture_path output/textured0.png --images_dir output/images
    
    print(f"Using Point Cloud: {point_cloud_path}")
    print(f"Using Sparse Model: {sparse_model_path}")
    print(f"Using Images: {images_dir}")
    
    if not os.path.exists(point_cloud_path):
        print(f"Error: Point cloud not found at {point_cloud_path}")
        if not args.reconstruct:
            print("Try running with --reconstruct to generate it from input/video.mp4")
        return

    # Load cameras from COLMAP
    try:
        cameras = load_cameras_from_colmap(sparse_model_path)
    except Exception as e:
        print(f"Error loading cameras from {sparse_model_path}: {e}")
        return
    
    config = FloorPlanGeneratorConfig.default(
        camera_intrinsics=cameras[0]['intrinsics'], 
        image_size=(cameras[0]['image_width'], cameras[0]['image_height']),
        texture_path=texture_path
    )

    # Initialize YOLO model
    yolo_model = YOLO('models/yolo/yolo11n.pt')
    yolo_detector = YOLODetector(yolo_model, config.detection)
    
    generator = FloorPlanGenerator(config, yolo_detector)

    # Match images with cameras based on image names
    # Build a mapping of image names to camera data
    camera_dict = {cam['image_name']: cam for cam in cameras}
    
    # Load only images that have corresponding camera data
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))
    matched_images = []
    matched_cameras = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)  # Get filename
        
        if img_name in camera_dict:
            img = cv2.imread(img_path)
            if img is not None:
                matched_images.append(img)
                matched_cameras.append(camera_dict[img_name])
                print(f"Matched: {img_name}")
        else:
            print(f"Skipping {img_name} (no camera data)")
    
    images = matched_images
    cameras = matched_cameras
    
    print(f"\nLoaded {len(images)} images with camera data")
    print(f"Skipped {len(image_files) - len(images)} images without camera data")

    if len(images) == 0:
        print("Error: No images with matching camera data found!")
        return
    
    try:
        results = generator.run_pipeline(
            point_cloud_path=point_cloud_path,
            images=images,
            cameras=cameras,
            output_path=floor_plan_output
        )
        print("Pipeline completed successfully!")
    except Exception as e:
        traceback.print_exc()
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run()