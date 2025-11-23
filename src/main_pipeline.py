import numpy as np
from typing import List, Dict, Any
import open3d as o3d
import glob
import cv2
from ultralytics import YOLO
import pycolmap

from config import FloorPlanGeneratorConfig
from point_cloud_processor import PointCloudProcessor
from furniture_clusterer import FurnitureClusterer
from label_projector import LabelProjector
from segmentation import StructuralSegmenter
from visualizer import FloorPlanVisualizer
from yolo_wrapper import YOLODetector
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
        point_cloud, alignment_rotation = self._align_reconstruction(point_cloud, cameras)
        
        processed_pcd = self.point_cloud_processor.preprocess(point_cloud)

        # Detect furniture in 2d
        detections = self.yolo_detector.detect_furniture(images)

        segmentation_results = self.segmenter.segment(processed_pcd)
        furniture_points = segmentation_results.furniture_points

        clusters = self.furniture_clusterer.cluster_furniture_points(furniture_points)

        labeled_clusters = self.label_projector.project_labels_to_clusters(
            clusters, detections, cameras
        )

        self.floor_plan_visualizer.generate_floor_plan(
            labeled_clusters,
            mesh_path=self.config.floor_plan.mesh_path,
            texture_path=self.config.floor_plan.texture_path,
            output_path=output_path
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


def load_cameras_from_colmap(sparse_path: str) -> List[Dict[str, Any]]:
    """
    Load camera parameters from COLMAP sparse reconstruction.
    
    Args:
        sparse_path: Path to COLMAP sparse directory (e.g., 'test/sparse/0')
    
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
    POINT_CLOUD_PATH = "test/point_cloud.ply"
    SPARSE_PATH = "test/sparse/0"
    OUTPUT_PATH = "test/floor_plan.png"
    TEXTURE_PATH = "test/textured0.png"
    
    # Load cameras from COLMAP
    cameras = load_cameras_from_colmap(SPARSE_PATH)
    
    config = FloorPlanGeneratorConfig.default(
        camera_intrinsics=cameras[0]['intrinsics'], 
        image_size=(cameras[0]['image_width'], cameras[0]['image_height']),
        texture_path=TEXTURE_PATH
    )

    # Initialize YOLO model
    yolo_model = YOLO('models/yolo/yolo11n.pt')
    yolo_detector = YOLODetector(yolo_model, config.detection)
    
    generator = FloorPlanGenerator(config, yolo_detector)

    # Match images with cameras based on image names
    # Build a mapping of image names to camera data
    camera_dict = {cam['image_name']: cam for cam in cameras}
    
    # Load only images that have corresponding camera data
    image_files = sorted(glob.glob("test/images/*.jpg") + glob.glob("test/images/*.png"))
    matched_images = []
    matched_cameras = []
    
    for img_path in image_files:
        img_name = img_path.split('\\')[-1].split('/')[-1]  # Get filename
        
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
            point_cloud_path=POINT_CLOUD_PATH,
            images=images,
            cameras=cameras,
            output_path=OUTPUT_PATH
        )
        print("Pipeline completed successfully!")
    except Exception as e:
        traceback.print_exc()
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run()