import open3d as o3d
import numpy as np
from typing import Dict, List, Tuple, Any
from config import SegmentationConfig


class StructuralSegmenter:
    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment(self, point_cloud: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Segment point cloud into floor, walls, ceiling, and furniture
        """
        points = np.asarray(point_cloud.points)

        print("Detecting floor plane")
        floor_plane, floor_indices = self.detect_floor(point_cloud)

        print("Detecting ceiling")
        _, ceil_indices = self.detect_ceil(point_cloud, floor_indices)

        print("Detecting walls")
        wall_indices = self.detect_walls(point_cloud, floor_indices, ceil_indices)

        # Remaining points are potential furniture
        all_indices = set(range(len(points)))
        structural = set(floor_indices) | set(ceil_indices) | set(wall_indices)
        furniture_indices = list(all_indices - structural)

        print(f"Segmentation: floor={len(floor_indices)}, walls={len(wall_indices)}, "
              f"ceiling={len(ceil_indices)}, furniture={len(furniture_indices)}")


        return {
            'floor_points': point_cloud.select_by_index(floor_indices),
            'wall_points': point_cloud.select_by_index(wall_indices),
            'ceiling_points': point_cloud.select_by_index(ceil_indices),
            'furniture_points': point_cloud.select_by_index(furniture_indices),
            'floor_plane': floor_plane,
            'floor_indices': floor_indices,
            'furniture_indices': furniture_indices,
        }

    def detect_floor(self, point_cloud: o3d.geometry.PointCloud) -> Tuple[np.ndarray, List[int]]:
        """Use RANSAC"""
        best_floor = None
        best_inliers = []
        max_inliers = 0

        tmp_pcd = point_cloud
        if len(tmp_pcd.points) < 100:
           raise Exception("too few points")

        for attempt in range(3):
            plane_model, inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.floor_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            if len(inliers) <= max_inliers:
                continue

            # Basic equation: a*x + b*y + c*z + d = 0
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

            # Check if horizontal
            vertical_dot = np.abs(normal[1])
            if vertical_dot > self.config.floor_horizontal_threshold:
                # Inverse plane equation if needed
                if normal[1] < 0:
                    normal = -normal
                    plane_model = -plane_model
                    a, b, c, d = plane_model

                # Calculate signed distance from each point to the possible plane
                distances = (a * tmp_pcd[:, 0] +
                             b * tmp_pcd[:, 1] +
                             c * tmp_pcd[:, 2] + d)
                points_above = np.sum(distances > 0)

                # For the floor most other points should be above it.
                if points_above > len(tmp_pcd) * 0.5:
                    if len(inliers) > max_inliers:
                        best_floor = plane_model
                        best_inliers = inliers
                        max_inliers = len(inliers)

            # Remove detected plane and try again
            remaining = list(set(range(len(tmp_pcd.points))) - set(inliers))
            tmp_pcd = tmp_pcd.select_by_index(remaining)

        if best_floor is None:
            raise Exception("Could not detect floor plane")

        print(f"Floor plane: {best_floor}, inliers: {len(best_inliers)}")
        return best_floor, best_inliers


    def detect_ceil(self, pcd: o3d.geometry.PointCloud,
                    floor_indices: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Use RANSAC"""
        points = np.asarray(pcd.points)
        best_ceiling = None
        best_inliers = []
        max_inliers = 0

        # Remove floor points first
        all_indices = set(range(len(points)))
        indeces = list(all_indices - set(floor_indices))

        if len(indeces) < 100:
            raise Exception("too few points")

        tmp_pcd = pcd.select_by_index(indeces)

        for attempt in range(3):
            if len(tmp_pcd.points) < 100:
                break

            plane_model, inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.ceil_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            a, b, c, d = plane_model

            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

            vertical_dot = np.abs(normal[1])
            if vertical_dot > self.config.ceil_horizontal_threshold:
                # For ceiling, normal should point downward
                if normal[1] > 0:
                    normal = -normal
                    plane_model = -plane_model
                    a, b, c, d = plane_model

                distances = (a * tmp_pcd[:, 0] +
                             b * tmp_pcd[:, 1] +
                             c * tmp_pcd[:, 2] + d)
                points_below = np.sum(distances > 0)

                if points_below > len(tmp_pcd) * 0.5:
                    if len(inliers) > max_inliers:
                        best_ceiling = plane_model
                        best_inliers = inliers
                        max_inliers = len(inliers)

            # Remove the detected plane and continue searching
            remaining_indices = list(set(range(len(tmp_pcd.points))) - set(inliers))
            tmp_pcd = tmp_pcd.select_by_index(remaining_indices)

        if best_ceiling is None:
            raise Exception("Could not detect ceil plane")

        print(f"Ceil plane: {best_ceiling}, inliers: {len(best_inliers)}")
        return best_ceiling, best_inliers


    def detect_walls(self, pcd: o3d.geometry.PointCloud,
                     floor_indices: List[int], ceil_indices: List[int]) -> List[int]:
        """Use RANSAC"""
        points = np.asarray(pcd.points)
        all_indices = set(range(len(points)))
        floor_and_ceil  = set(floor_indices) | set(ceil_indices)
        available_indices = list(all_indices - floor_and_ceil)

        if len(available_indices) < 100:
            raise Exception("too few points")

        tmp_pcd = pcd.select_by_index(available_indices)
        wall_indices = []

        min_wall_points = max(100, len(tmp_pcd.points) * self.config.min_wall_points_ratio)

        for wall_num in range(self.config.max_walls):
            if len(tmp_pcd.points) < min_wall_points:
                break

            plane_model, inliers = tmp_pcd.segment_plane(
                distance_threshold=self.config.wall_distance_threshold,
                ransac_n=self.config.plane_ransac_n,
                num_iterations=self.config.plane_ransac_iterations
            )

            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

            # Check if vertical
            horizontal_component = np.abs(normal[1])

            if horizontal_component < self.config.wall_vertical_threshold and len(inliers) > min_wall_points:
                wall_indices.extend(inliers)

                available_indices = [available_indices[i] for i in range(len(available_indices))
                                     if i not in inliers]
                print(f"Wall {wall_num + 1}: {len(inliers)} points")

            remaining_indices_local = list(set(range(len(tmp_pcd.points))) - set(inliers))
            if len(remaining_indices_local) == 0:
                break

            tmp_pcd = tmp_pcd.select_by_index(remaining_indices_local)

        return wall_indices