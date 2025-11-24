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
                            texture_path: str = None, 
                            output_path: str = "floor_plan.png",
                            alignment_matrix: Optional[np.ndarray] = None,
                            wall_planes: Optional[List[np.ndarray]] = None) -> None:
        """
        Generate 2D top-down view with labeled furniture bounding boxes and room background
        """
        print("Generating 2D floor plan with background...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Render and draw background mesh
        mesh_extent = None
        if os.path.exists(mesh_path):
            bg_img, mesh_extent = self._render_top_down_view(mesh_path, texture_path, alignment_matrix, wall_planes)
            if bg_img is not None:
                # origin='lower' places (0,0) at bottom-left
                ax.imshow(bg_img, extent=mesh_extent, origin='lower', alpha=1.0)
                if self.config.debug:
                    print(f"Rendered mesh extent (X, Z): {mesh_extent}")
        else:
            print(f"Warning: Mesh not found at {mesh_path}")

        # 2. Draw furniture clusters
        clusters_outside_extent = 0
        for cluster in clusters:
            # Validate cluster is within mesh extent
            if mesh_extent is not None:
                in_extent = self._validate_cluster_in_extent(cluster, mesh_extent)
                if not in_extent:
                    clusters_outside_extent += 1
                    if self.config.debug:
                        print(f"⚠️  Cluster '{cluster.label}' extends outside mesh extent")
            
            self._draw_cluster(ax, cluster)
        
        if clusters_outside_extent > 0:
            print(f"⚠️  WARNING: {clusters_outside_extent} cluster(s) extend outside mesh render extent")
            print(f"   This may indicate alignment issues. Check debug output for details.")

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

    def _render_top_down_view(self, mesh_path: str, texture_path: str = None, 
                              alignment_matrix: Optional[np.ndarray] = None,
                              wall_planes: Optional[List[np.ndarray]] = None) -> Tuple[Optional[np.ndarray], List[float]]:
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

            # Apply alignment if provided
            if alignment_matrix is not None:
                print("Applying alignment matrix to mesh...")
                if self.config.debug:
                    print(f"Alignment matrix:\n{alignment_matrix}")
                    print(f"Vertex 0 before: {tm_mesh.vertices[0]}")
                
                # Convert 3x3 rotation to 4x4 transform
                transform = np.eye(4)
                transform[:3, :3] = alignment_matrix
                tm_mesh.apply_transform(transform)
                
                if self.config.debug:
                    print(f"Vertex 0 after: {tm_mesh.vertices[0]}")

            # Remove walls if provided
            if wall_planes:
                print(f"Removing {len(wall_planes)} detected walls from mesh...")
                vertices = tm_mesh.vertices
                
                # A point (x,y,z) is on plane (a,b,c,d) if ax+by+cz+d = 0
                # Distance is |ax+by+cz+d| / sqrt(a^2+b^2+c^2)
                # Since plane models from Open3D are usually normalized (a^2+b^2+c^2=1), distance is just |ax+by+cz+d|
                
                # We want to remove faces that have vertices close to any wall plane
                # Let's be aggressive: if any vertex of a face is close to a wall, remove the face
                
                # Calculate distances to all planes
                # planes: (N_planes, 4)
                # vertices: (N_verts, 3)
                # We need to append 1 to vertices for dot product
                
                ones = np.ones((len(vertices), 1))
                hom_vertices = np.hstack((vertices, ones)) # (N_verts, 4)
                
                planes_matrix = np.array(wall_planes).T # (4, N_planes)
                
                # Distances (signed)
                distances = np.dot(hom_vertices, planes_matrix) # (N_verts, N_planes)
                
                # Check if close to any wall (e.g. within 10cm)
                wall_threshold = 0.10 
                is_wall_vertex = np.any(np.abs(distances) < wall_threshold, axis=1)
                
                # Filter faces
                # Keep faces where NO vertex is a wall vertex
                # faces: (N_faces, 3) indices
                
                face_wall_mask = is_wall_vertex[tm_mesh.faces] # (N_faces, 3) boolean
                # If any vertex is a wall, remove face
                faces_to_remove = np.any(face_wall_mask, axis=1)
                
                print(f"Removing {np.sum(faces_to_remove)} faces belonging to walls")
                
                tm_mesh.update_faces(~faces_to_remove)
                tm_mesh.remove_unreferenced_vertices()

            # 2. CROP (Geometry first)
            # We calculate the crop based on aligned vertices
            vertices = tm_mesh.vertices
            y_values = vertices[:, 1]
            y_min, y_max = y_values.min(), y_values.max()
            
            # Apply configurable crop ratio (default 1.0 = no cropping)
            crop_ratio = self.config.mesh_height_crop_ratio
            cutoff = y_min + ((y_max - y_min) * crop_ratio)
            
            if self.config.debug:
                print(f"Mesh Y range: [{y_min:.3f}, {y_max:.3f}]")
                print(f"Crop ratio: {crop_ratio:.2f} (cutoff at Y={cutoff:.3f})")
            
            # Filter faces
            face_mask = (y_values[tm_mesh.faces] <= cutoff).all(axis=1)
            removed_faces = len(tm_mesh.faces) - np.sum(face_mask)
            if removed_faces > 0:
                print(f"Removed {removed_faces} faces above Y={cutoff:.3f} (crop ratio: {crop_ratio:.2f})")
            
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

            if pil_img is None and texture_path:
                if os.path.exists(texture_path):
                    import PIL.Image
                    pil_img = PIL.Image.open(texture_path)
            
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
            min_b = aabb.get_min_bound()
            max_b = aabb.get_max_bound()
            center = aabb.get_center()
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=2048, height=2048)
            vis.add_geometry(mesh)
            
            opt = vis.get_render_option()
            opt.light_on = True 
            opt.mesh_show_back_face = True

            ctr = vis.get_view_control()
            ctr.set_front([0, -1, 0])
            ctr.set_lookat(center)
            ctr.set_up([0, 0, 1])
            
            # Switch to orthographic view for correct floor plan projection
            ctr.change_field_of_view(step=-90)
            
            # --- CALIBRATION STEP ---
            # Render with a distinct background color to measure the exact pixel footprint of the mesh.
            # This allows us to calculate the precise pixels-per-world-unit ratio.
            calibration_bg = np.asarray([1.0, 0.0, 1.0]) # Magenta
            opt.background_color = calibration_bg
            
            for _ in range(5):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)
                
            calib_image = vis.capture_screen_float_buffer(do_render=True)
            calib_np = np.asarray(calib_image)
            
            # Find bounding box of non-background pixels
            # Check for pixels that are NOT magenta (allow small tolerance for anti-aliasing/compression)
            mask = np.any(np.abs(calib_np - calibration_bg) > 0.01, axis=2)
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                print("⚠️  Warning: Could not detect mesh in rendered image for calibration. Using default scaling.")
                # Fallback to AABB-based extent (assuming fit is tight, which is usually wrong but best guess)
                range_x = max_b[0] - min_b[0]
                range_z = max_b[2] - min_b[2]
                max_span = max(range_x, range_z)
                extent_size = max_span
            else:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                
                pixel_width = xmax - xmin
                pixel_height = ymax - ymin
                
                world_width = max_b[0] - min_b[0]
                world_height = max_b[2] - min_b[2]
                
                # Calculate pixels per world unit
                # We use the average of X and Z ratios for robustness, though they should be identical in orthographic
                ppu_x = pixel_width / world_width if world_width > 0 else 0
                ppu_z = pixel_height / world_height if world_height > 0 else 0
                
                if ppu_x == 0 or ppu_z == 0:
                     ppu = max(ppu_x, ppu_z)
                else:
                     ppu = (ppu_x + ppu_z) / 2
                
                if ppu <= 0:
                    print("⚠️  Warning: Invalid PPU calculated. Using fallback.")
                    extent_size = max(world_width, world_height)
                else:
                    # Calculate the total world size covered by the 2048x2048 image
                    extent_size = 2048 / ppu
                    if self.config.debug:
                        print(f"Calibration: Mesh World Size: {world_width:.2f}x{world_height:.2f}")
                        print(f"Calibration: Mesh Pixel Size: {pixel_width}x{pixel_height}")
                        print(f"Calibration: PPU: {ppu:.2f} -> Extent Size: {extent_size:.2f}")

            # Calculate final extent centered on the mesh center
            extent = [
                center[0] - extent_size / 2,
                center[0] + extent_size / 2,
                center[2] - extent_size / 2,
                center[2] + extent_size / 2
            ]

            # --- FINAL RENDER ---
            # Switch background to white for the final output
            opt.background_color = np.asarray([1.0, 1.0, 1.0])
            vis.update_renderer()
            time.sleep(0.05)
            
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            image_np = np.asarray(image)
            
            # Flip horizontally because Camera Right (-X) is opposite to Plot Right (+X)
            image_np = np.fliplr(image_np)
            # Flip vertically because Open3D Row 0 is Top (+Z), but imshow origin='lower' expects Row 0 at Bottom (-Z)
            image_np = np.flipud(image_np)
            
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

    def _validate_cluster_in_extent(self, cluster: FurnitureCluster, extent: List[float]) -> bool:
        """
        Check if cluster bounding box is within rendered mesh extent.
        extent = [x_min, x_max, z_min, z_max]
        Returns True if cluster is fully within bounds, False otherwise.
        """
        bbox = cluster.bbox_3d
        x_min = bbox.left_down_corner[0]
        x_max = bbox.right_up_corner[0]
        z_min = bbox.left_down_corner[2]
        z_max = bbox.right_up_corner[2]
        
        # Check if cluster is within extent
        within_x = extent[0] <= x_min and x_max <= extent[1]
        within_z = extent[2] <= z_min and z_max <= extent[3]
        
        return within_x and within_z

    def _add_legend(self, ax, clusters: List[FurnitureCluster]):
        from matplotlib.lines import Line2D
        present_labels = set(c.label for c in clusters)
        legend_elements = [
            Line2D([0], [0], color=self.color_map[l], lw=4, label=l)
            for l in self.color_map if l in present_labels
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')