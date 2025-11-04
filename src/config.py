from typing import List
import numpy as np


class DetectionConfig:
    def __init__(self, min_confidence: float = 0.5, furniture_classes: List[str] = None):
        self.min_confidence = min_confidence
        self.furniture_classes = furniture_classes if furniture_classes is not None else [
            'chair', 'couch', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'keyboard', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'potted plant'
        ]


class PointCloudConfig:
    def __init__(self, statistical_outlier_neighbors: int = 30, statistical_outlier_std_ratio: float = 2.0,
                 radius_outlier_neighbors: int = 16, radius_outlier_radius: float = 0.05,
                 normal_search_radius: float = 0.1, normal_max_nn: int = 30):
        self.statistical_outlier_neighbors = statistical_outlier_neighbors
        self.statistical_outlier_std_ratio = statistical_outlier_std_ratio
        self.radius_outlier_neighbors = radius_outlier_neighbors
        self.radius_outlier_radius = radius_outlier_radius
        self.normal_search_radius = normal_search_radius
        self.normal_max_nn = normal_max_nn


class SegmentationConfig:
    def __init__(self,
                 floor_distance_threshold: float = 0.02,
                 ceil_distance_threshold: float = 0.02,
                 wall_distance_threshold: float = 0.03,

                 plane_ransac_n: int = 3, plane_ransac_iterations: int = 1000,

                 floor_horizontal_threshold: float = 0.85,
                 ceil_horizontal_threshold: float = 0.85,
                 wall_vertical_threshold: float = 0.3,

                 max_walls: int = 6, min_wall_points_ratio: float = 0.03):

        self.floor_distance_threshold = floor_distance_threshold
        self.ceil_distance_threshold = ceil_distance_threshold
        self.wall_distance_threshold = wall_distance_threshold

        self.plane_ransac_n = plane_ransac_n
        self.plane_ransac_iterations = plane_ransac_iterations

        self.floor_horizontal_threshold = floor_horizontal_threshold
        self.ceil_horizontal_threshold = ceil_horizontal_threshold
        self.wall_vertical_threshold = wall_vertical_threshold

        self.max_walls = max_walls
        self.min_wall_points_ratio = min_wall_points_ratio


class ClusteringConfig:
    def __init__(self, dbscan_eps: float = 0.15, dbscan_min_samples: int = 20,
                 min_furniture_points: int = 50, max_furniture_height: float = 3.0,
                 use_adaptive_eps: bool = True, eps_dense: float = 0.05,
                 eps_medium: float = 0.10, eps_sparse: float = 0.15):
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_furniture_points = min_furniture_points
        self.max_furniture_height = max_furniture_height
        self.use_adaptive_eps = use_adaptive_eps
        self.eps_dense = eps_dense
        self.eps_medium = eps_medium
        self.eps_sparse = eps_sparse


class LabelingConfig:
    def __init__(self, n_sample_points: int = 20, min_projection_distance: float = 0.1,
                 min_vote_count: int = 2, min_vote_score: float = 1.5,
                 distance_weight_scale: float = 5.0, bbox_size_penalty: float = 0.5):
        self.n_sample_points = n_sample_points
        self.min_projection_distance = min_projection_distance
        self.min_vote_count = min_vote_count
        self.min_vote_score = min_vote_score
        self.distance_weight_scale = distance_weight_scale
        self.bbox_size_penalty = bbox_size_penalty


class FloorPlanConfig:
    def __init__(self, min_object_confidence: float = 0.5, simplify_footprint_threshold: int = 8,
                 units: str = "meters", coordinate_system: str = "XZ (Y-up, floor at Y=0)"):
        self.min_object_confidence = min_object_confidence
        self.simplify_footprint_threshold = simplify_footprint_threshold
        self.units = units
        self.coordinate_system = coordinate_system


class CameraConfig:
    def __init__(self, intrinsics: np.ndarray, image_width: int, image_height: int):
        self.intrinsics = np.array(intrinsics)
        self.image_width = image_width
        self.image_height = image_height


class FloorPlanGeneratorConfig:
    def __init__(self, detection: DetectionConfig = None, point_cloud: PointCloudConfig = None,
                 segmentation: SegmentationConfig = None, clustering: ClusteringConfig = None,
                 labeling: LabelingConfig = None, floor_plan: FloorPlanConfig = None,
                 camera: CameraConfig = None):
        self.detection = detection if detection is not None else DetectionConfig()
        self.point_cloud = point_cloud if point_cloud is not None else PointCloudConfig()
        self.segmentation = segmentation if segmentation is not None else SegmentationConfig()
        self.clustering = clustering if clustering is not None else ClusteringConfig()
        self.labeling = labeling if labeling is not None else LabelingConfig()
        self.floor_plan = floor_plan if floor_plan is not None else FloorPlanConfig()
        self.camera = camera

    @classmethod
    def default(cls, camera_intrinsics, image_size):
        return cls(
            camera=CameraConfig(
                intrinsics=camera_intrinsics,
                image_width=image_size[0],
                image_height=image_size[1]
            )
        )