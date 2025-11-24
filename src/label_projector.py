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

        # Merge overlapping clusters
        merged_clusters = self._merge_overlapping_clusters(labeled_clusters)
        
        # Filter out small unknown clusters
        final_clusters = self._filter_clusters(merged_clusters)

        return final_clusters

    def project_labels_to_points(self,
                                 points: np.ndarray,
                                 detections: List[List[Dict[str, Any]]],
                                 cameras: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, int]]]:
        """
        Project 2D YOLO detections to individual 3D points.
        Returns:
            point_labels: List of assigned labels for each point
            point_votes: List of vote dictionaries for each point
        """
        print(f"Projecting 2D labels to {len(points)} 3D points...")
        
        num_points = len(points)
        # Initialize votes for each point
        point_votes = [defaultdict(float) for _ in range(num_points)]
        
        num_views = len(detections)
        
        for view_idx in range(num_views):
            camera_detections = detections[view_idx]
            camera_info = cameras[view_idx]
            
            if not camera_detections:
                continue
                
            print(f"Processing view {view_idx + 1}/{num_views} with {len(camera_detections)} detections")
            
            # Project all points to this camera
            u, v, depth = self._project_points_to_camera(points, camera_info)
            
            # Filter valid projections
            valid_mask = (depth > 0)
            
            if not np.any(valid_mask):
                continue
                
            # For each detection, find points inside bbox
            for detection in camera_detections:
                bbox_2d = detection['bbox'] # [x1, y1, x2, y2]
                label = detection['label']
                confidence = detection['confidence']
                
                # Vectorized check for points inside bbox
                in_bbox_mask = (u >= bbox_2d[0]) & (u <= bbox_2d[2]) & \
                               (v >= bbox_2d[1]) & (v <= bbox_2d[3]) & \
                               valid_mask
                               
                # Indices of points inside bbox
                point_indices = np.where(in_bbox_mask)[0]
                
                if len(point_indices) == 0:
                    continue
                    
                # Calculate weights
                # closer points get higher weight
                depths = depth[point_indices]
                weights = confidence * np.exp(-depths / self.config.distance_weight_scale)
                
                # Update votes
                for idx, weight in zip(point_indices, weights):
                    point_votes[idx][label] += weight

        # Assign final labels
        point_labels = []
        labeled_count = 0
        
        for votes in point_votes:
            if not votes:
                point_labels.append('unknown')
                continue
                
            # Find best label
            best_label = max(votes.items(), key=lambda x: x[1])[0]
            best_score = votes[best_label]
            
            # Use a threshold for acceptance
            if best_score > self.config.min_vote_score: 
                point_labels.append(best_label)
                labeled_count += 1
            else:
                point_labels.append('unknown')
                
        print(f"Labeled {labeled_count} out of {num_points} points")
        return point_labels, point_votes

    def _merge_overlapping_clusters(self, clusters: List[FurnitureCluster]) -> List[FurnitureCluster]:
        """Merge overlapping clusters with compatible labels"""
        print(f"Merging overlapping clusters (initial: {len(clusters)})")
        from furniture_clusterer import BBox3d  # Import here to avoid circular dependency
        
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
        # Check overlap
        if not self._check_overlap(c1.bbox_3d, c2.bbox_3d):
            return False
            
        # Check labels
        l1 = c1.label
        l2 = c2.label
        
        # Always merge same labels
        if l1 == l2:
            return True
            
        # Merge unknown into known
        if l1 == 'unknown' or l2 == 'unknown':
            return True
            
        # Different known labels - don't merge
        return False

    def _check_overlap(self, bbox1, bbox2) -> bool:
        # Check if bboxes intersect
        # Check X
        if bbox1.right_up_corner[0] < bbox2.left_down_corner[0] or \
           bbox2.right_up_corner[0] < bbox1.left_down_corner[0]:
            return False
            
        # Check Y (height)
        if bbox1.right_up_corner[1] < bbox2.left_down_corner[1] or \
           bbox2.right_up_corner[1] < bbox1.left_down_corner[1]:
            return False

        # Check Z
        if bbox1.right_up_corner[2] < bbox2.left_down_corner[2] or \
           bbox2.right_up_corner[2] < bbox1.left_down_corner[2]:
            return False
            
        return True

    def _merge_clusters(self, c1: FurnitureCluster, c2: FurnitureCluster) -> FurnitureCluster:
        from furniture_clusterer import BBox3d
        
        # Combine points
        new_points = np.vstack((c1.points, c2.points))
        
        # Combine votes
        new_votes = c1.label_votes.copy()
        if c2.label_votes:
            for label, count in c2.label_votes.items():
                new_votes[label] += count
            
        # Determine new label
        new_label = c1.label
        if c1.label == 'unknown' and c2.label != 'unknown':
            new_label = c2.label
        
        # Recompute bbox
        min_coords = np.min(new_points, axis=0)
        max_coords = np.max(new_points, axis=0)
        center = (min_coords + max_coords) / 2
        
        new_bbox = BBox3d(min_coords, max_coords, center)
        
        return FurnitureCluster(
            points=new_points,
            cluster_id=c1.cluster_id,
            bbox_3d=new_bbox,
            num_points=len(new_points),
            label=new_label,
            label_votes=new_votes
        )

    def _filter_clusters(self, clusters: List[FurnitureCluster]) -> List[FurnitureCluster]:
        """Filter out noise clusters"""
        filtered = []
        for c in clusters:
            # Remove unknown clusters that are likely noise
            if c.label == 'unknown':
                # If it's very small, drop it
                if c.num_points < 100:
                    continue
            filtered.append(c)
        
        print(f"Filtered {len(clusters) - len(filtered)} noise clusters")
        return filtered

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

    def _project_points_to_camera(self, points: np.ndarray, camera_info: Dict[str, Any]) -> tuple:
        """
        Project multiple 3D points to 2D image coordinates.
        Returns: u, v, depth (arrays of shape (N,))
        """
        intrinsics = camera_info.get('intrinsics')
        rotation = camera_info.get('rotation')
        translation = camera_info.get('translation')
        image_width = camera_info.get('image_width', 640)
        image_height = camera_info.get('image_height', 480)

        if intrinsics is None or rotation is None or translation is None:
            return np.zeros(len(points)), np.zeros(len(points)), -np.ones(len(points))

        # Transform points to camera coordinates
        # points: (N, 3)
        # rotation: (3, 3)
        # translation: (3,)
        # point_cam = R @ p + t => points @ R.T + t
        
        points_cam = points @ rotation.T + translation
        
        # Extract coordinates
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Project to image plane
        # Avoid division by zero
        z_safe = np.where(z <= 0, 1e-6, z)
        
        u_norm = x / z_safe
        v_norm = y / z_safe
        
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        u = fx * u_norm + cx
        v = fy * v_norm + cy
        
        # Mark invalid points
        # Behind camera
        invalid_mask = (z <= 0)
        # Outside image
        invalid_mask |= (u < 0) | (u >= image_width) | (v < 0) | (v >= image_height)
        
        # Set depth of invalid points to -1
        depth = z.copy()
        depth[invalid_mask] = -1
        
        return u, v, depth

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