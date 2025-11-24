import numpy as np
import open3d as o3d
import glob
import cv2
import sys
import os
from typing import List, Dict, Any
import matplotlib.colors as mcolors

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import FloorPlanGeneratorConfig
from main_pipeline import FloorPlanGenerator, load_cameras_from_colmap
from yolo_wrapper import YOLODetector
from ultralytics import YOLO

def visualize_3d_results():
    # Paths
    POINT_CLOUD_PATH = "output/point_cloud.ply"
    SPARSE_PATH = "output/sparse/0"
    TEXTURE_PATH = "output/textured0.png"
    MESH_PATH = "output/final.ply"
    
    if not os.path.exists(POINT_CLOUD_PATH):
        print(f"Error: Point cloud not found at {POINT_CLOUD_PATH}")
        return

    # 1. Setup Config and Generator
    print("Setting up pipeline...")
    try:
        cameras = load_cameras_from_colmap(SPARSE_PATH)
    except Exception as e:
        print(f"Error loading cameras: {e}")
        return
    
    config = FloorPlanGeneratorConfig.default(
        camera_intrinsics=cameras[0]['intrinsics'], 
        image_size=(cameras[0]['image_width'], cameras[0]['image_height']),
        texture_path=TEXTURE_PATH
    )
    # Ensure mesh path is correct in config
    config.floor_plan.mesh_path = MESH_PATH

    print("Loading YOLO model...")
    yolo_model = YOLO('models/yolo/yolo11n.pt')
    yolo_detector = YOLODetector(yolo_model, config.detection)
    
    generator = FloorPlanGenerator(config, yolo_detector)

    # 2. Load Images and Match Cameras
    print("Loading images...")
    camera_dict = {cam['image_name']: cam for cam in cameras}
    image_files = sorted(glob.glob("output/images/*.jpg") + glob.glob("output/images/*.png"))
    matched_images = []
    matched_cameras = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        if img_name in camera_dict:
            img = cv2.imread(img_path)
            if img is not None:
                matched_images.append(img)
                matched_cameras.append(camera_dict[img_name])
    
    images = matched_images
    cameras = matched_cameras
    
    if not images:
        print("No images found!")
        return
    print(f"Loaded {len(images)} images.")

    # 3. Run Pipeline Steps (Manually to get intermediate results)
    print("Loading point cloud...")
    point_cloud = o3d.io.read_point_cloud(POINT_CLOUD_PATH)
    
    # Alignment
    print("Aligning reconstruction...")
    point_cloud, R_cam = generator._align_reconstruction(point_cloud, cameras)
    point_cloud, R_floor = generator._align_to_floor(point_cloud, cameras)
    point_cloud, R_walls = generator._align_to_walls(point_cloud, cameras)
    alignment_rotation = R_walls @ R_floor @ R_cam
    
    # Processing
    print("Preprocessing point cloud...")
    processed_pcd = generator.point_cloud_processor.preprocess(point_cloud)
    
    # Detection
    print("Running YOLO detection...")
    detections = generator.yolo_detector.detect_furniture(images)
    
    # Segmentation
    print("Segmenting point cloud...")
    segmentation_results = generator.segmenter.segment(processed_pcd)
    furniture_points = segmentation_results.furniture_points
    
    # New Pipeline: Label Points -> Filter -> Cluster
    print("Projecting labels to points...")
    furniture_points_np = np.asarray(furniture_points.points)
    point_labels, _ = generator.label_projector.project_labels_to_points(
        furniture_points_np, detections, cameras
    )
    
    # Filter out unknown points
    print("Filtering unknown points...")
    valid_indices = [i for i, label in enumerate(point_labels) if label != 'unknown']
    
    if not valid_indices:
        print("No furniture points labeled. Skipping clustering.")
        labeled_clusters = []
    else:
        cleaned_points = furniture_points_np[valid_indices]
        cleaned_labels = [point_labels[i] for i in valid_indices]
        
        print(f"Kept {len(cleaned_points)} points after filtering unknown labels")
        
        # Clustering
        print("Clustering labeled points...")
        labeled_clusters = generator.furniture_clusterer.cluster_labeled_points(
            cleaned_points, cleaned_labels
        )
    
    # 4. Visualize in 3D
    print("\n--- Starting 3D Visualization ---")
    visualize_clusters_and_mesh(labeled_clusters, MESH_PATH, alignment_rotation, segmentation_results.wall_planes)

def visualize_clusters_and_mesh(clusters, mesh_path, alignment_matrix, wall_planes=None):
    geometries = []
    
    # Load Mesh
    if os.path.exists(mesh_path):
        print(f"Loading mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # Apply alignment
        transform = np.eye(4)
        transform[:3, :3] = alignment_matrix
        mesh.transform(transform)
        
        # Remove walls if provided
        if wall_planes:
            print(f"Removing {len(wall_planes)} detected walls from 3D mesh...")
            vertices = np.asarray(mesh.vertices)
            
            # Calculate distances to all planes
            ones = np.ones((len(vertices), 1))
            hom_vertices = np.hstack((vertices, ones))
            planes_matrix = np.array(wall_planes).T
            distances = np.dot(hom_vertices, planes_matrix)
            
            wall_threshold = 0.10
            is_wall_vertex = np.any(np.abs(distances) < wall_threshold, axis=1)
            
            # Filter triangles
            triangles = np.asarray(mesh.triangles)
            # Check if any vertex of the triangle is a wall vertex
            triangle_wall_mask = is_wall_vertex[triangles]
            triangles_to_remove = np.any(triangle_wall_mask, axis=1)
            
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            print(f"Removed {np.sum(triangles_to_remove)} triangles belonging to walls")

        geometries.append(mesh)
    else:
        print(f"Warning: Mesh not found at {mesh_path}")

    # Create Bounding Boxes
    # Generate distinct colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    unique_labels = sorted(list(set(c.label for c in clusters)))
    label_color_map = {label: mcolors.to_rgb(colors[i % len(colors)]) for i, label in enumerate(unique_labels)}
    
    print("Detected Objects:")
    for cluster in clusters:
        if cluster.label == 'unknown':
            color = [0.5, 0.5, 0.5] # Grey for unknown
        else:
            color = label_color_map[cluster.label]
            
        bbox = cluster.bbox_3d
        # Open3D AxisAlignedBoundingBox
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox.left_down_corner,
            max_bound=bbox.right_up_corner
        )
        aabb.color = color
        geometries.append(aabb)
        
        # Also add a point cloud of the cluster points to see the density
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster.points)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)
        
        print(f"  - {cluster.label} at {bbox.center}")

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    print("Opening Open3D window... (Close window to exit)")
    
    # Try using the new visualization API which supports transparency and line width better
    if hasattr(o3d.visualization, 'draw'):
        print("Using new Open3D visualization API (supports transparency)")
        draw_geometries = []
        
        # Add room mesh
        if geometries and isinstance(geometries[0], o3d.geometry.TriangleMesh):
             draw_geometries.append(geometries[0])
             
        # Add coordinate frame
        draw_geometries.append(coord_frame)
        
        for cluster in clusters:
            if cluster.label == 'unknown':
                color = [0.5, 0.5, 0.5]
            else:
                color = label_color_map[cluster.label]
            
            bbox = cluster.bbox_3d
            
            # 1. Box Mesh (Transparent Faces)
            width = bbox.right_up_corner[0] - bbox.left_down_corner[0]
            height = bbox.right_up_corner[1] - bbox.left_down_corner[1]
            depth = bbox.right_up_corner[2] - bbox.left_down_corner[2]
            
            box_mesh = o3d.geometry.TriangleMesh.create_box(width, height, depth)
            box_mesh.translate(bbox.left_down_corner)
            box_mesh.paint_uniform_color(color)
            
            mat_box = o3d.visualization.rendering.MaterialRecord()
            mat_box.shader = "defaultLitTransparency"
            mat_box.base_color = [*color, 0.25] # 25% opacity
            
            draw_geometries.append({'name': f"cluster_{cluster.cluster_id}_box", 'geometry': box_mesh, 'material': mat_box})
            
            # 2. Edges (Thick Lines)
            lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                o3d.geometry.AxisAlignedBoundingBox(bbox.left_down_corner, bbox.right_up_corner)
            )
            lines.paint_uniform_color(color)
            
            mat_line = o3d.visualization.rendering.MaterialRecord()
            mat_line.shader = "unlitLine"
            mat_line.line_width = 8.0 # Thicker lines
            
            draw_geometries.append({'name': f"cluster_{cluster.cluster_id}_lines", 'geometry': lines, 'material': mat_line})
            
            # 3. Points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster.points)
            pcd.paint_uniform_color(color)
            
            mat_pcd = o3d.visualization.rendering.MaterialRecord()
            mat_pcd.shader = "defaultLit"
            mat_pcd.point_size = 6.0
            
            draw_geometries.append({'name': f"cluster_{cluster.cluster_id}_pcd", 'geometry': pcd, 'material': mat_pcd})

        o3d.visualization.draw(draw_geometries)
        return

    # Fallback for older Open3D versions
    geometries.append(coord_frame)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Visualization", width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    opt = vis.get_render_option()
    opt.line_width = 10.0
    opt.point_size = 5.0
    opt.background_color = np.asarray([1, 1, 1])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_results()
