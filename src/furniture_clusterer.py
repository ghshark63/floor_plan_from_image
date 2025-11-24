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
        # if len(points) > 1000000:
        #     print(f"Downsampling from {len(points)} points to reduce memory usage...")
        #     indices = np.random.choice(len(points), 1000000, replace=False)
        #     points = points[indices]
        #     print(f"Downsampled to {len(points)} points")

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
                print(f"Detected box is too tall: height={height:.2f}")
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

    def cluster_labeled_points(self, points: np.ndarray, point_labels: List[str]) -> List[FurnitureCluster]:
        """
        Cluster points that have already been labeled.
        """
        print(f"Clustering {len(points)} labeled points...")
        
        if len(points) == 0:
            return []

        # Determine appropriate eps
        eps = self._get_adaptive_eps(points) if self.config.use_adaptive_eps else self.config.dbscan_eps
        print(f"DBSCAN eps={eps:.3f}")
        
        clustering = DBSCAN(eps=eps, min_samples=self.config.dbscan_min_samples).fit(points)
        labels = clustering.labels_
        
        unique_cluster_ids = set(labels)
        unique_cluster_ids.discard(-1)
        
        clusters = []
        
        for cluster_id in unique_cluster_ids:
            mask = (labels == cluster_id)
            cluster_points = points[mask]
            cluster_point_labels = np.array(point_labels)[mask]
            
            # Filter by size
            if len(cluster_points) < self.config.min_furniture_points:
                continue
                
            # Determine label
            # Count labels in this cluster
            unique, counts = np.unique(cluster_point_labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
            
            # Remove 'unknown' from voting if present
            if 'unknown' in label_counts:
                del label_counts['unknown']
            
            if not label_counts:
                continue
                
            best_label = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # Create cluster
            bbox = self._compute_bbox(cluster_points)
            
            # Check height
            height = bbox.right_up_corner[1] - bbox.left_down_corner[1]
            if height > self.config.max_furniture_height:
                continue
                
            # Create votes dict for the cluster
            votes = defaultdict(int)
            for l, c in label_counts.items():
                votes[l] = int(c)
            
            clusters.append(
                FurnitureCluster(
                    points=cluster_points,
                    cluster_id=cluster_id,
                    bbox_3d=bbox,
                    num_points=len(cluster_points),
                    label=best_label,
                    label_votes=votes
                )
            )
            
        print(f"Created {len(clusters)} clusters from labeled points")
        
        # Merge overlapping clusters with same label
        clusters = self.merge_overlapping_clusters(clusters)
        
        return clusters

    def merge_overlapping_clusters(self, clusters: List[FurnitureCluster]) -> List[FurnitureCluster]:
        """Merge overlapping clusters with compatible labels"""
        print(f"Merging overlapping clusters (initial: {len(clusters)})")
        
        merged = True
        while merged:
            merged = False
            new_clusters = []
            skip_indices = set()
            
            # Sort by number of points (descending) to merge smaller into larger
            clusters.sort(key=lambda c: c.num_points, reverse=True)
            
            for i in range(len(clusters)):
                if i in skip_indices:
                    continue
                
                current_cluster = clusters[i]
                
                for j in range(i + 1, len(clusters)):
                    if j in skip_indices:
                        continue
                    
                    other_cluster = clusters[j]
                    
                    if self._should_merge(current_cluster, other_cluster):
                        # Merge j into i
                        current_cluster = self._merge_clusters(current_cluster, other_cluster)
                        skip_indices.add(j)
                        merged = True
                
                new_clusters.append(current_cluster)
            
            clusters = new_clusters
            if merged:
                print(f"  Merged down to {len(clusters)} clusters")
        
        return clusters

    def _should_merge(self, c1: FurnitureCluster, c2: FurnitureCluster) -> bool:
        # Check overlap with some padding to be aggressive (0.75m padding)
        if not self._check_overlap(c1.bbox_3d, c2.bbox_3d, padding=0.75):
            return False
            
        # Check labels
        l1 = c1.label
        l2 = c2.label
        
        # Always merge same labels
        if l1 == l2:
            return True
            
        return False

    def _check_overlap(self, bbox1: BBox3d, bbox2: BBox3d, padding: float = 0.0) -> bool:
        # Check if bboxes intersect with padding
        # Check X
        if bbox1.right_up_corner[0] + padding < bbox2.left_down_corner[0] - padding or \
           bbox2.right_up_corner[0] + padding < bbox1.left_down_corner[0] - padding:
            return False
            
        # Check Y (height)
        if bbox1.right_up_corner[1] + padding < bbox2.left_down_corner[1] - padding or \
           bbox2.right_up_corner[1] + padding < bbox1.left_down_corner[1] - padding:
            return False

        # Check Z
        if bbox1.right_up_corner[2] + padding < bbox2.left_down_corner[2] - padding or \
           bbox2.right_up_corner[2] + padding < bbox1.left_down_corner[2] - padding:
            return False
            
        return True

    def _merge_clusters(self, c1: FurnitureCluster, c2: FurnitureCluster) -> FurnitureCluster:
        # Combine points
        new_points = np.vstack((c1.points, c2.points))
        
        # Combine votes if they exist
        new_votes = defaultdict(int)
        if c1.label_votes:
            for l, c in c1.label_votes.items():
                new_votes[l] += c
        if c2.label_votes:
            for l, c in c2.label_votes.items():
                new_votes[l] += c
            
        # Determine new label (should be same)
        new_label = c1.label
        
        # Recompute bbox
        min_coords = np.min(new_points, axis=0)
        max_coords = np.max(new_points, axis=0)
        center = (min_coords + max_coords) / 2
        
        new_bbox = BBox3d(min_coords, max_coords, center)
        
        return FurnitureCluster(
            points=new_points,
            cluster_id=c1.cluster_id, # Keep ID of larger cluster
            bbox_3d=new_bbox,
            num_points=len(new_points),
            label=new_label,
            label_votes=new_votes
        )