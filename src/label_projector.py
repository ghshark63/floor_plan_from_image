import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from config import LabelingConfig
from furniture_clusterer import FurnitureCluster


class LabelProjector:
    # Reference data:
    # https://colmap.github.io/format.html#images-txt
    # https://github.com/colmap/colmap/issues/1594
    # https://github.com/colmap/colmap/issues/1424
    # https://github.com/colmap/colmap/issues/1476
    def __init__(self, config: LabelingConfig):
        self.config = config

    def project_labels_to_clusters(self,
                                   clusters: List[FurnitureCluster],
                                   detections: List[List[Dict[str, Any]]],
                                   cameras: List[Dict[str, Any]]) -> List[FurnitureCluster]:
        """
        Project 2D YOLO detections to 3D clusters using multi-view voting
        """
        print("Projecting 2D labels to 3D clusters")

        # Initialize label votes for each cluster
        for cluster in clusters:
            cluster.label_votes = defaultdict(int)

        num_views = len(detections)

        # For each camera view and its detections
        for view_idx in range(num_views):
            camera_detections = detections[view_idx]
            camera_info = cameras[view_idx]

            if not camera_detections:
                continue

            print(f"Processing view {view_idx + 1}/{num_views} with {len(camera_detections)} detections")

            for cluster in clusters:
                self._project_cluster_to_camera(cluster, camera_detections, camera_info)

        # Assign final labels based on voting
        labeled_clusters = self._assign_final_labels(clusters)

        return labeled_clusters

    def _project_cluster_to_camera(self,
                                   cluster: FurnitureCluster,
                                   detections: List[Dict[str, Any]],
                                   camera_info: Dict[str, Any]):
        """Project a cluster to a camera view and check for detection matches"""
        cluster_points = cluster.points
        bbox_3d = cluster.bbox_3d

        # Sample points from the cluster for projection
        n_samples = min(self.config.n_sample_points, len(cluster_points))

        sample_indices = np.random.choice(len(cluster_points), n_samples, replace=False)
        sample_points = cluster_points[sample_indices]

        # Project sample points to image
        projected_points = []
        for point in sample_points:
            u, v, depth = self._project_point_to_camera(point, camera_info)
            if u is not None and depth > 0:
                projected_points.append((u, v, depth))

        if not projected_points:
            return

        # Check each detection for matches
        for detection in detections:
            bbox_2d = detection['bbox']  # [x1, y1, x2, y2]
            label = detection['label']
            confidence = detection['confidence']

            # Count how many projected points fall within this detection
            points_in_detection = 0
            total_confidence = 0

            for u, v, depth in projected_points:
                if (bbox_2d[0] <= u <= bbox_2d[2] and
                        bbox_2d[1] <= v <= bbox_2d[3]):
                    points_in_detection += 1
                    # Weight by confidence and distance (closer = higher weight)
                    distance_weight = np.exp(-depth / self.config.distance_weight_scale)
                    total_confidence += confidence * distance_weight

            if points_in_detection > 0:
                cluster.label_votes[label] += 1

    def _project_point_to_camera(self, point: np.ndarray, camera_info: Dict[str, Any]) -> tuple:
        """Project 3D point to 2D image coordinates"""
        intrinsics = camera_info.get('intrinsics')
        rotation = camera_info.get('rotation')
        translation = camera_info.get('translation')
        image_width = camera_info.get('image_width', 640)
        image_height = camera_info.get('image_height', 480)

        if intrinsics is None or rotation is None or translation is None:
            return None, None, -1

        # Transform point to camera coordinates
        point_cam = rotation @ point + translation

        # Check if point is behind camera
        if point_cam[2] <= 0:
            return None, None, -1

        # Project to image plane
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]

        u = intrinsics[0, 0] * x + intrinsics[0, 2]
        v = intrinsics[1, 1] * y + intrinsics[1, 2]

        # Check if within image bounds
        if 0 <= u < image_width and 0 <= v < image_height:
            return u, v, point_cam[2]
        else:
            return None, None, -1

    def _assign_final_labels(self, clusters: List[FurnitureCluster]) -> List[FurnitureCluster]:
        """Assign final labels to clusters based on voting results"""
        labeled_count = 0

        for cluster in clusters:
            best_label = 'unknown'
            best_votes = 0

            if cluster.label_votes is None or len(cluster.label_votes) == 0:
                cluster.label = 'unknown'
                continue

            for label, votes in cluster.label_votes.items():
                if votes < self.config.min_vote_count:
                    continue

                if votes > best_votes:
                    best_votes = votes
                    best_label = label

            cluster.label = best_label

            if best_label != 'unknown':
                labeled_count += 1
                print(f"Cluster {cluster.cluster_id}: assigned {best_label}, "
                      f"votes: {best_votes})")

        print(f"Successfully labeled {labeled_count} of {len(clusters)} clusters")
        return clusters