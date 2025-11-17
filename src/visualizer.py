import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict
from config import FloorPlanConfig
from furniture_clusterer import FurnitureCluster


class FloorPlanVisualizer:
    def __init__(self, config: FloorPlanConfig):
        self.config = config
        self.color_map = self._generate_color_map()

    def generate_floor_plan(self, clusters: List[FurnitureCluster],
                            output_path: str = "floor_plan.png") -> None:
        """
        Generate 2D top-down view with labeled furniture bounding boxes
        """
        print("Generating 2D floor plan")

        fig, ax = plt.subplots(figsize=(12, 10))


        for cluster in clusters:
            self._draw_cluster(ax, cluster)

        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('2D Floor Plan')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        self._add_legend(ax, clusters)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Floor plan saved to {output_path}")
        plt.show()

    def _generate_color_map(self) -> Dict[str, str]:
        """Generate a color map with random RGB colors for furniture classes"""
        import random

        # Use the furniture classes from config to assign colors
        from config import DetectionConfig  # Import here to avoid circular imports
        detection_config = DetectionConfig()
        furniture_classes = detection_config.furniture_classes + ['unknown']

        color_map = {}
        for furniture_class in furniture_classes:
            # Generate random RGB values
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            color_map[furniture_class] = (r, g, b)

        return color_map

    def _draw_cluster(self, ax, cluster: FurnitureCluster):
        """Draw a single cluster on the floor plan"""
        bbox_3d = cluster.bbox_3d
        label = cluster.label

        min_x = bbox_3d.left_down_corner[0]
        min_z = bbox_3d.left_down_corner[2]
        max_x = bbox_3d.right_up_corner[0]
        max_z = bbox_3d.right_up_corner[2]
        width = max_x - min_x
        height = max_z - min_z

        color = self.color_map[label]

        # Draw bounding box
        rect = patches.Rectangle(
            (min_x, min_z), width, height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

        label_text = label
        ax.text(
            min_x + width/2, min_z + height/2, label_text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            ha='center', va='center', fontsize=8, color='white', weight='bold'
        )

        # Add center point
        ax.plot(min_x + width/2, min_z + height/2, 'o', color=color, markersize=3)

    def _add_legend(self, ax, clusters: List[FurnitureCluster]):
        from matplotlib.lines import Line2D

        present_labels = set()
        for cluster in clusters:
            label = cluster.label
            present_labels.add(label)

        legend_elements = []
        for label, color in self.color_map.items():
            if label in present_labels:
                legend_elements.append(
                    Line2D([0], [0], color=color, lw=4, label=label)
                )

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')