import numpy as np
import torch
from typing import List, Dict
from config import DetectionConfig


class YOLODetector:
    def __init__(self, yolo_model, config: DetectionConfig):
        self.model = yolo_model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLODetector using device: {self.device}")

    def detect_furniture(self, images: List[np.ndarray]) -> List[List[Dict]]:
        all_detections = []

        for img_idx, img in enumerate(images):
            results = self.model(img, verbose=False, device=self.device)
            image_detections = self._parse_results(results[0], img_idx)
            all_detections.append(image_detections)

            print(f"Image {img_idx}: detected {len(image_detections)} furnitures")

        return all_detections

    def _parse_results(self, results, img_idx: int) -> List[Dict]:
        detections = []

        if not hasattr(results, 'boxes'):
            raise Exception("no boxes attr in yolo results")

        boxes = results.boxes

        for box in boxes:
            confidence = float(box.conf.item())
            if confidence < self.config.min_confidence:
                continue

            class_id = int(box.cls.item())
            label = self.model.names[class_id]

            if not self._is_furniture(label):
                print(f"label {label} is not in config known labels")
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # TODO: remove mask
            # Get segmentation mask if available (YOLOv8-seg)
            mask = None
            if hasattr(box, 'masks') and box.masks is not None:
                mask = box.masks.data.cpu().numpy()

            detections.append({
                'label': label.lower(),
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': class_id,
                'mask': mask,
                'image_idx': img_idx,
                'bbox_area': float((x2 - x1) * (y2 - y1))
            })

        return detections

    def _is_furniture(self, label: str) -> bool:
        return any(item.lower() in label.lower()
                   for item in self.config.furniture_classes)