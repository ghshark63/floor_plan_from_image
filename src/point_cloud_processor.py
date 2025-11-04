import open3d as o3d
from config import PointCloudConfig


class PointCloudProcessor:
    def __init__(self, config: PointCloudConfig):
        self.config = config

    def preprocess(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        _, inds = point_cloud.remove_statistical_outlier(
            nb_neighbors=self.config.statistical_outlier_neighbors,
            std_ratio=self.config.statistical_outlier_std_ratio
        )
        clean_pcd = point_cloud.select_by_index(inds)


        _, inds = clean_pcd.remove_radius_outlier(
            nb_points=self.config.radius_outlier_neighbors,
            radius=self.config.radius_outlier_radius
        )
        clean_pcd = clean_pcd.select_by_index(inds)

        if not clean_pcd.has_normals():
            print("Computing normals")
            clean_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_search_radius,
                    max_nn=self.config.normal_max_nn
                )
            )
            clean_pcd.orient_normals_consistent_tangent_plane(k=15)

        return clean_pcd
