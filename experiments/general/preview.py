import open3d as o3d
import argparse
import os


def preview_ply(filename: str):
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return

    # Try reading as mesh
    mesh = o3d.io.read_triangle_mesh(filename)
    if mesh and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"Displaying mesh: {filename}")
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
        return

    # If no triangles, try showing as point cloud
    pcd = o3d.io.read_point_cloud(filename)
    if pcd and len(pcd.points) > 0:
        print(f"Displaying point cloud: {filename}")
        o3d.visualization.draw_geometries([pcd])
        return

    print(f"❌ Failed to load a valid mesh or point cloud from: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview a .ply 3D model with Open3D"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the .ply file you want to preview",
    )

    args = parser.parse_args()
    preview_ply(args.model)
