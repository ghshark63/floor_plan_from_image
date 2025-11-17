import numpy as np
from typing import List, Dict, Any
import open3d as o3d
import glob
import cv2
from ultralytics import YOLO

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

def run():
    # example

    cameras = [
        {
            'intrinsics': np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]),
            'rotation': np.eye(3),
            'translation': np.array([0, 0, 0]),
            'image_width': 640,
            'image_height': 480
        }
    ]
    config = FloorPlanGeneratorConfig.default(camera_intrinsics=cameras[0]['intrinsics'], image_size=(640, 480))

    # Initialize YOLO model
    yolo_model = YOLO('models/yolo/yolo11n.pt')
    yolo_detector = YOLODetector(yolo_model, config.detection)
    
    generator = FloorPlanGenerator(config, yolo_detector)

    # Load images from /test/images/
    image_files = glob.glob("test/images/*.jpg")
    images = [cv2.imread(f) for f in image_files]

    try:
        results = generator.run_pipeline(
            point_cloud_path="test/point_cloud.ply",
            images=images,
            cameras=cameras,
            output_path="floor_plan.png"
        )
        print("Pipeline completed successfully!")
    except Exception as e:
        traceback.print_exc()
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run()