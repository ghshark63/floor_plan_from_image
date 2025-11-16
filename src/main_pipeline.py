import numpy as np
from typing import List, Dict, Any

from config import FloorPlanGeneratorConfig
from point_cloud_processor import PointCloudProcessor
from furniture_clusterer import FurnitureClusterer
from label_projector import LabelProjector
from src.segmentation import StructuralSegmenter
from src.visualizer import FloorPlanVisualizer


class FloorPlanGenerator:
    def __init__(self, config: FloorPlanGeneratorConfig):
        self.config = config
        self.yolo_detector = None
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
        point_cloud = ...
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
    config = FloorPlanGeneratorConfig.default()

    generator = FloorPlanGenerator(config)

    # yolo stuff ...

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

    images = ...

    try:
        results = generator.run_pipeline(
            point_cloud_path="path/to/pointcloud.ply",
            images=images,
            cameras=cameras,
            output_path="floor_plan.png"
        )
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run()