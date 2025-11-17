from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict
from config import ClusteringConfig
from open3d.cpu.pybind.geometry import  PointCloud



@dataclass
class BBox3d:
    left_down_corner: np.ndarray
    right_up_corner: np.ndarray
    center: np.ndarray

@dataclass
class FurnitureCluster:
    points: np.ndarray
    cluster_id: int
    bbox_3d: BBox3d
    num_points: int

    # Will be populated later
    label: str = "unknown"
    label_votes: Dict[str, int] | None = None # label to number of votes

class FurnitureClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster_furniture_points(self, furniture_points: PointCloud) -> List[FurnitureCluster]:
        """
        Connect furniture points into individual objects using DBSCAN
        """
        points = np.asarray(furniture_points.points)
        
        print(f"Initial furniture points: {len(points)}")
        
        # Downsample if too many points to avoid memory issues
        if len(points) > 1000000:
            print(f"Downsampling from {len(points)} points to reduce memory usage...")
            indices = np.random.choice(len(points), 1000000, replace=False)
            points = points[indices]
            print(f"Downsampled to {len(points)} points")

        # Determine appropriate eps based on point density if adaptive eps is enabled
        eps = self._get_adaptive_eps(points) if self.config.use_adaptive_eps else self.config.dbscan_eps

        print(f"Clustering {len(points)} furniture points with DBSCAN (eps={eps:.3f}, min_samples={self.config.dbscan_min_samples})")

        # DBSCAN in sklearn should work for 3d space
        # https://stackoverflow.com/questions/26246015/python-dbscan-in-3-dimensional-space
        clustering = DBSCAN(eps=eps, min_samples=self.config.dbscan_min_samples).fit(points)


        labels = clustering.labels_

        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise

        print(f"Found {len(unique_labels)} potential furniture clusters")

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]

            # Filter by size constraints
            if len(cluster_points) < self.config.min_furniture_points:
                continue

            bbox = self._compute_bbox(cluster_points)
            height = bbox.right_up_corner[1] - bbox.left_down_corner[1]
            if height > self.config.max_furniture_height:
                print("Detected box is too small")
                continue

            clusters.append(
                FurnitureCluster(
                    points=cluster_points,
                    cluster_id=label,
                    bbox_3d=bbox,
                    num_points=len(cluster_points)
                )
            )

        print(f"Found {len(clusters)} furniture clusters after filtering")
        return clusters

    def _get_adaptive_eps(self, points: np.ndarray) -> float:
        """Determine eps based on point density"""
        from sklearn.neighbors import NearestNeighbors

        if len(points) < 100:
            return self.config.eps_sparse

        # Compute average distance to k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points))).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_distances = np.mean(distances, axis=1)
        median_distance = np.median(avg_distances)

        if median_distance < 0.03:
            return self.config.eps_dense
        elif median_distance < 0.08:
            return self.config.eps_medium
        else:
            return self.config.eps_sparse

    def _compute_bbox(self, points: np.ndarray) -> BBox3d:
        """Compute 3d bounding box for a cluster"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2

        return BBox3d(
            left_down_corner=min_coords,
            right_up_corner=max_coords,
            center=center
        )