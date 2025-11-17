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
        processed_pcd = self.point_cloud_processor.preprocess(point_cloud)

        # Detect furniture in 2d
        detections = self.yolo_detector.detect_furniture(images)

        segmentation_results = self.segmenter.segment(processed_pcd)
        furniture_points = segmentation_results.furniture_points

        clusters = self.furniture_clusterer.cluster_furniture_points(furniture_points)

        labeled_clusters = self.label_projector.project_labels_to_clusters(
            clusters, detections, cameras
        )

        self.floor_plan_visualizer.generate_floor_plan(labeled_clusters, output_path)


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
    
    # Load cameras from COLMAP
    cameras = load_cameras_from_colmap(SPARSE_PATH)
    
    config = FloorPlanGeneratorConfig.default(
        camera_intrinsics=cameras[0]['intrinsics'], 
        image_size=(cameras[0]['image_width'], cameras[0]['image_height'])
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