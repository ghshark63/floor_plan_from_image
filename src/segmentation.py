from dataclasses import dataclass

import numpy as np
from typing import List
from config import SegmentationConfig
from open3d.cpu.pybind.geometry import PointCloud

@dataclass
class SegmentationResult:
    # Point clouds
    floor_points: PointCloud
    wall_points: PointCloud
    ceiling_points: PointCloud
    furniture_points: PointCloud

    # Indices relative to original point cloud
    floor_indices: List[int]
    wall_indices: List[int]
    ceil_indices: List[int]
    furniture_indices: List[int]

class StructuralSegmenter:
    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment(self, point_cloud: PointCloud) -> SegmentationResult:
        """
        Segment point cloud into floor, walls, ceiling, and furniture
        """
        points = np.asarray(point_cloud.points)

        print("Detecting floor plane")
        floor_indices = self.detect_floor(point_cloud)

        print("Detecting ceiling")
        ceil_indices = self.detect_ceil(point_cloud, floor_indices)

        print("Detecting walls")
        wall_indices = self.detect_walls(point_cloud, floor_indices, ceil_indices)

        # Remaining points are potential furniture
        all_indices = set(range(len(points)))
        structural = set(floor_indices) | set(ceil_indices) | set(wall_indices)
        furniture_indices = list(all_indices - structural)

        print(f"Segmentation: floor={len(floor_indices)}, walls={len(wall_indices)}, "
              f"ceiling={len(ceil_indices)}, furniture={len(furniture_indices)}")

        return SegmentationResult(
            floor_points=point_cloud.select_by_index(floor_indices),
            wall_points=point_cloud.select_by_index(wall_indices),
            ceiling_points=point_cloud.select_by_index(ceil_indices),
            furniture_points=point_cloud.select_by_index(furniture_indices),
            floor_indices=floor_indices,
            wall_indices=wall_indices,
            ceil_indices=ceil_indices,
            furniture_indices=furniture_indices,
        )

    def detect_floor(self, point_cloud: PointCloud) -> List[int]:
        if len(point_cloud.points) < 100:
            raise Exception("Too few points for floor detection")

        points_array = np.asarray(point_cloud.points)
        best_global_inliers = []
        max_inliers = 0

        available_global_indices = list(range(len(point_cloud.points)))
        tmp_pcd: PointCloud = point_cloud

        for attempt in range(3):
            plane_model, local_inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.floor_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            print(f"  Floor attempt {attempt + 1}: found {len(local_inliers)} inliers")
            
            if len(local_inliers) > max_inliers:
                # Convert local inliers to global inliers
                global_inliers = [available_global_indices[i] for i in local_inliers]

                # Basic equation: a*x + b*y + c*z + d = 0
                a, b, c, d = plane_model
                normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
                print(f"  Plane normal: {normal}, vertical component: {np.abs(normal[1]):.3f}")

                # Check if horizontal
                vertical_dot = np.abs(normal[1])
                if vertical_dot > self.config.floor_horizontal_threshold:

                    # Floor should be lower than majority of the points
                    inlier_points = points_array[global_inliers]
                    if len(inlier_points) > 0:
                        y_values = inlier_points[:, 1]
                        median_y = np.median(y_values)
                        percentile_25 = np.percentile(points_array[:, 1], 25)
                        print(f"  Floor plane median Y: {median_y:.3f}, 25th percentile: {percentile_25:.3f}")
                        if median_y < percentile_25:
                            best_global_inliers = global_inliers
                            max_inliers = len(global_inliers)
                            print(f"  ✓ Accepted as floor plane")
                        else:
                            print(f"  ✗ Rejected: not at bottom of point cloud")
                    else:
                        print(f"  ✗ Rejected: plane is horizontal but position check failed")
                else:
                    print(f"  ✗ Rejected: not horizontal enough (threshold: {self.config.floor_horizontal_threshold})")

            # Remove detected plane and try again
            remaining_local_indices = list(set(range(len(tmp_pcd.points))) - set(local_inliers))

            tmp_pcd = tmp_pcd.select_by_index(remaining_local_indices)
            available_global_indices = [available_global_indices[i] for i in remaining_local_indices]

        print(f"Number of inliers for floor: {len(best_global_inliers)}")
        return best_global_inliers


    def detect_ceil(self, point_cloud: PointCloud,
                    floor_indices: List[int]) -> List[int]:
        points = np.asarray(point_cloud.points)

        # Remove floor points first
        all_indices = set(range(len(points)))
        available_global_indices = list(all_indices - set(floor_indices))

        if len(available_global_indices) < 100:
            raise Exception("Too few points for ceiling detection")

        tmp_pcd: PointCloud = point_cloud.select_by_index(available_global_indices)
        points_array = np.asarray(tmp_pcd.points)

        best_global_inliers = []
        max_inliers = 0

        for attempt in range(3):
            plane_model, local_inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.ceil_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            print(f"  Ceiling attempt {attempt + 1}: found {len(local_inliers)} inliers")
            
            if len(local_inliers) > max_inliers:
                # Convert local inliers to global inliers
                global_inliers = [available_global_indices[i] for i in local_inliers]

                a, b, c, d = plane_model
                normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
                print(f"  Plane normal: {normal}, vertical component: {np.abs(normal[1]):.3f}")

                # Check if horizontal
                vertical_dot = np.abs(normal[1])

                if vertical_dot > self.config.ceil_horizontal_threshold:
                    # Ceiling should be near the top
                    inlier_points = points_array[local_inliers]
                    if len(inlier_points) > 0:
                        y_values = inlier_points[:, 1]
                        median_y = np.median(y_values)
                        percentile_75 = np.percentile(points_array[:, 1], 75)
                        print(f"  Ceiling plane median Y: {median_y:.3f}, 75th percentile: {percentile_75:.3f}")
                        if median_y > percentile_75:
                            best_global_inliers = global_inliers
                            max_inliers = len(global_inliers)
                            print(f"  ✓ Accepted as ceiling plane")
                        else:
                            print(f"  ✗ Rejected: not at top of point cloud")
                    else:
                        print(f"  ✗ Rejected: plane is horizontal but position check failed")
                else:
                    print(f"  ✗ Rejected: not horizontal enough (threshold: {self.config.ceil_horizontal_threshold})")

            # Remove detected plane and continue
            remaining_local_indices = list(set(range(len(tmp_pcd.points))) - set(local_inliers))

            tmp_pcd = tmp_pcd.select_by_index(remaining_local_indices)
            available_global_indices = [available_global_indices[i] for i in remaining_local_indices]

        print(f"Number of inliers for ceil: {len(best_global_inliers)}")
        return best_global_inliers

    def detect_walls(self, pcd: PointCloud,
                     floor_indices: List[int], ceil_indices: List[int]) -> List[int]:
        points = np.asarray(pcd.points)
        all_indices = set(range(len(points)))
        floor_and_ceil = set(floor_indices) | set(ceil_indices)
        available_global_indices = list(all_indices - floor_and_ceil)

        if len(available_global_indices) < 100:
            return []

        tmp_pcd: PointCloud = pcd.select_by_index(available_global_indices)
        wall_global_indices = []
        min_wall_points = int(len(tmp_pcd.points) * self.config.min_wall_points_ratio)

        for wall_num in range(self.config.max_walls):
            if len(tmp_pcd.points) < min_wall_points:
                break

            plane_model, local_inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.wall_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            print(f"  Wall attempt {wall_num + 1}: found {len(local_inliers)} inliers")
            
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            horizontal_component = np.abs(normal[1])
            print(f"  Plane normal: {normal}, horizontal component: {horizontal_component:.3f}")

            # Check if vertical
            if horizontal_component < self.config.wall_vertical_threshold:
                global_inliers = [available_global_indices[i] for i in local_inliers]
                wall_global_indices.extend(global_inliers)
                print(f"Wall {wall_num + 1}: {len(global_inliers)} points")

            remaining_local_indices = list(set(range(len(tmp_pcd.points))) - set(local_inliers))

            tmp_pcd = tmp_pcd.select_by_index(remaining_local_indices)
            available_global_indices = [available_global_indices[i] for i in remaining_local_indices]

        return wall_global_indices
