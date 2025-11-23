import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from config import FloorPlanConfig
from furniture_clusterer import FurnitureCluster

class FloorPlanVisualizer:
    def __init__(self, config: FloorPlanConfig):
        self.config = config
        self.color_map = self._generate_color_map()

    def generate_floor_plan(self, clusters: List[FurnitureCluster],
                            mesh_path: str, 
                            output_path: str = "floor_plan.png") -> None:
        """
        Generate 2D top-down view with labeled furniture bounding boxes and room background
        """
        print("Generating 2D floor plan with background...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Render and draw background mesh
        if os.path.exists(mesh_path):
            bg_img, extent = self._render_top_down_view(mesh_path)
            if bg_img is not None:
                # origin='lower' places (0,0) at bottom-left
                ax.imshow(bg_img, extent=extent, origin='lower', alpha=1.0)
        else:
            print(f"Warning: Mesh not found at {mesh_path}")

        # 2. Draw furniture clusters
        for cluster in clusters:
            self._draw_cluster(ax, cluster)

        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('2D Floor Plan')
        ax.grid(True, alpha=0.3)
        
        # Ensure the plot aspect ratio matches the physical dimensions
        ax.set_aspect('equal')

        self._add_legend(ax, clusters)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Floor plan saved to {output_path}")
        plt.show()

    def _render_top_down_view(self, mesh_path: str) -> Tuple[Optional[np.ndarray], List[float]]:
        import trimesh
        import numpy as np
        import open3d as o3d
        import os
        import time

        try:
            print("Rendering: Explicit Triangle Soup + UV V-Flip...")

            # 1. Load mesh with Trimesh (let it handle the header parsing)
            tm_mesh = trimesh.load(mesh_path, process=True)
            
            if isinstance(tm_mesh, trimesh.Scene):
                if len(tm_mesh.geometry) == 0: return None, []
                tm_mesh = tm_mesh.dump(concatenate=True)

            # 2. CROP (Geometry first)
            # We calculate the crop based on original vertices
            vertices = tm_mesh.vertices
            y_values = vertices[:, 1]
            y_min, y_max = y_values.min(), y_values.max()
            cutoff = y_min + ((y_max - y_min) * 0.8)
            
            # Filter faces
            face_mask = (y_values[tm_mesh.faces] <= cutoff).all(axis=1)
            tm_mesh.update_faces(face_mask)
            tm_mesh.remove_unreferenced_vertices()

            # 3. CONSTRUCT TRIANGLE SOUP (The "Nuclear Option")
            # We completely explode the mesh so every triangle has its own 3 vertices.
            # This removes ANY ambiguity about shared UVs.
            
            # Get data from Trimesh
            old_verts = tm_mesh.vertices          # (N_v, 3)
            old_faces = tm_mesh.faces             # (N_f, 3)
            old_uvs = np.array(tm_mesh.visual.uv) # (N_v, 2) assuming process=True aligned them
            
            # Flatten: Create 3 vertices per face
            # This makes an array of shape (N_f * 3, 3)
            flat_verts = old_verts[old_faces.flatten()]
            
            # Flatten: Create 3 UVs per face
            flat_uvs = old_uvs[old_faces.flatten()]
            
            # *CRITICAL FIX*: Flip the Vertical (V) coordinate
            # OpenMVS UVs are Bottom-Left origin. Images are Top-Left origin.
            flat_uvs[:, 1] = 1.0 - flat_uvs[:, 1]

            # Create new faces: 0,1,2 / 3,4,5 / ...
            num_triangles = len(old_faces)
            flat_triangles = np.arange(num_triangles * 3).reshape((-1, 3))

            # 4. Create Open3D Mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(flat_verts)
            mesh.triangles = o3d.utility.Vector3iVector(flat_triangles)
            
            # Assign the flattened, flipped UVs
            mesh.triangle_uvs = o3d.utility.Vector2dVector(flat_uvs)
            
            # Recompute normals for shading
            mesh.compute_vertex_normals()

            # 5. Load Texture
            pil_img = None
            if hasattr(tm_mesh.visual, 'material') and hasattr(tm_mesh.visual.material, 'image'):
                pil_img = tm_mesh.visual.material.image

            if pil_img is None:
                mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
                tex_path = os.path.join(mesh_dir, "textured0.png")
                if os.path.exists(tex_path):
                    import PIL.Image
                    pil_img = PIL.Image.open(tex_path)

            if pil_img:
                img_np = np.asarray(pil_img)
                o3d_img = o3d.geometry.Image(img_np)
                mesh.textures = [o3d_img]
                # Use material ID 0 for all faces
                mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
            
            # 6. Render
            aabb = mesh.get_axis_aligned_bounding_box()
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=2048, height=2048)
            vis.add_geometry(mesh)
            
            opt = vis.get_render_option()
            opt.background_color = np.asarray([1.0, 1.0, 1.0]) 
            opt.light_on = True 
            opt.mesh_show_back_face = True

            ctr = vis.get_view_control()
            ctr.set_front([0, -1, 0])
            ctr.set_lookat(aabb.get_center())
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.45)
            
            for _ in range(5):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)
                
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            image_np = np.asarray(image)
            min_b = aabb.get_min_bound()
            max_b = aabb.get_max_bound()
            extent = [min_b[0], max_b[0], min_b[2], max_b[2]]
            
            return image_np, extent

        except Exception as e:
            print(f"Error rendering mesh: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    # ... existing _generate_color_map, _draw_cluster, _add_legend methods ...
    def _generate_color_map(self) -> Dict[str, str]:
        """Generate a color map with random RGB colors for furniture classes"""
        import random
        from config import DetectionConfig
        detection_config = DetectionConfig()
        furniture_classes = detection_config.furniture_classes + ['unknown']
        color_map = {}
        for furniture_class in furniture_classes:
            color_map[furniture_class] = (
                random.random(), random.random(), random.random()
            )
        return color_map

    def _draw_cluster(self, ax, cluster: FurnitureCluster):
        bbox_3d = cluster.bbox_3d
        label = cluster.label
        min_x, min_z = bbox_3d.left_down_corner[0], bbox_3d.left_down_corner[2]
        width = bbox_3d.right_up_corner[0] - min_x
        height = bbox_3d.right_up_corner[2] - min_z
        color = self.color_map[label]

        rect = patches.Rectangle(
            (min_x, min_z), width, height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(
            min_x + width/2, min_z + height/2, label,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            ha='center', va='center', fontsize=8, color='white', weight='bold'
        )
        ax.plot(min_x + width/2, min_z + height/2, 'o', color=color, markersize=3)

    def _add_legend(self, ax, clusters: List[FurnitureCluster]):
        from matplotlib.lines import Line2D
        present_labels = set(c.label for c in clusters)
        legend_elements = [
            Line2D([0], [0], color=self.color_map[l], lw=4, label=l)
            for l in self.color_map if l in present_labels
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')