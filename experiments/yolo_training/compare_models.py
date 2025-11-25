#!/usr/bin/env python3
import sys
import os
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path to allow importing from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import DetectionConfig

def setup_temp_coco_data(original_yaml_path, coco_model_names, temp_dir):
    """
    Creates a temporary dataset structure with labels mapped to COCO IDs.
    Returns the path to the new dataset.yaml and the list of relevant COCO IDs.
    """
    temp_dir = Path(temp_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # Load original dataset config
    with open(original_yaml_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    # Build mapping from Custom ID to COCO ID
    # dataset_cfg['names'] can be a list or dict
    custom_names = dataset_cfg['names']
    if isinstance(custom_names, list):
        custom_names = {i: n for i, n in enumerate(custom_names)}
    
    # Invert COCO names to map Name -> ID
    coco_name_to_id = {n: i for i, n in coco_model_names.items()}
    
    id_map = {} # Custom ID -> COCO ID
    relevant_coco_ids = []
    
    print("Mapping classes:")
    for cust_id, name in custom_names.items():
        if name in coco_name_to_id:
            coco_id = coco_name_to_id[name]
            id_map[cust_id] = coco_id
            relevant_coco_ids.append(coco_id)
            print(f"  {name}: Custom {cust_id} -> COCO {coco_id}")
        else:
            print(f"  Warning: Class '{name}' not found in COCO model classes. Skipping.")

    # Setup paths
    # Assuming val path is relative to yaml or absolute
    val_path_rel = dataset_cfg.get('val')
    original_val_dir = (Path(original_yaml_path).parent / val_path_rel).resolve()
    
    # We assume standard YOLO structure: images/val and labels/val
    # If original_val_dir points to images/val, labels should be in labels/val
    original_images_dir = original_val_dir
    original_labels_dir = original_val_dir.parent.parent / 'labels' / original_val_dir.name
    
    if not original_labels_dir.exists():
        # Try alternative: maybe labels are in the same folder or different structure
        # But for this project we saw data/coco_furniture/labels/val
        # and data/coco_furniture/images/val
        # So the above logic should work if val_path is ./images/val
        pass

    # Create temp structure
    temp_images_dir = temp_dir / 'images' / 'val'
    temp_labels_dir = temp_dir / 'labels' / 'val'
    temp_images_dir.mkdir(parents=True)
    temp_labels_dir.mkdir(parents=True)

    print(f"Preparing temporary validation data in {temp_dir}...")
    
    # Process files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for img_file in tqdm(list(original_images_dir.iterdir())):
        if img_file.suffix.lower() not in image_extensions:
            continue
            
        # Symlink image
        try:
            os.symlink(img_file, temp_images_dir / img_file.name)
        except OSError:
            shutil.copy(img_file, temp_images_dir / img_file.name)
            
        # Process label
        label_file = original_labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            new_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_id = int(parts[0])
                    
                    if cls_id in id_map:
                        new_cls_id = id_map[cls_id]
                        new_lines.append(f"{new_cls_id} {' '.join(parts[1:])}")
            
            if new_lines:
                with open(temp_labels_dir / label_file.name, 'w') as f:
                    f.write('\n'.join(new_lines))

    # Create temp dataset.yaml
    # We use the COCO names for the config, so the validator knows what 56 is.
    temp_yaml_content = {
        'path': str(temp_dir),
        'train': 'images/val', # Dummy
        'val': 'images/val',
        'names': coco_model_names
    }
    
    temp_yaml_path = temp_dir / 'dataset.yaml'
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_yaml_content, f)
        
    return temp_yaml_path, relevant_coco_ids

def main():
    # Paths
    models_dir = project_root / "models" / "yolo"
    # Check for the new finetuned model first
    finetuned_model_path = project_root / "experiments" / "yolo_training" / "innodorm_finetune" / "weights" / "best.pt"
    if not finetuned_model_path.exists():
        print(f"New finetuned model not found at {finetuned_model_path}, checking legacy path...")
        finetuned_model_path = models_dir / "yolo11finetuned.pt"
    
    regular_model_path = models_dir / "yolo11n.pt"
    dataset_yaml_path = project_root / "data" / "innodorm" / "dataset.yaml"
    
    if not finetuned_model_path.exists() or not regular_model_path.exists():
        print("Models not found.")
        return

    # 1. Evaluate Fine-tuned Model
    print("\n" + "="*50)
    print("Evaluating Fine-tuned Model")
    print("="*50)
    model_ft = YOLO(finetuned_model_path)
    try:
        results_ft = model_ft.val(
            data=str(dataset_yaml_path), 
            project=str(project_root / "experiments/yolo_training/runs"), 
            name="finetuned_eval",
            verbose=False
        )
        print(f"Fine-tuned mAP50:    {results_ft.box.map50:.4f}")
        print(f"Fine-tuned mAP50-95: {results_ft.box.map:.4f}")
    except Exception as e:
        print(f"Error evaluating fine-tuned model: {e}")

    # 2. Evaluate Regular Model (with class mapping)
    print("\n" + "="*50)
    print("Evaluating Regular Model (COCO Pre-trained)")
    print("="*50)
    
    model_reg = YOLO(regular_model_path)
    
    # Prepare temp data with COCO IDs
    temp_dir = project_root / "experiments" / "yolo_training" / "temp_coco_eval"
    try:
        temp_yaml_path, relevant_coco_ids = setup_temp_coco_data(
            dataset_yaml_path, 
            model_reg.names, 
            temp_dir
        )
        
        # Run validation on the temp dataset, filtering for relevant classes
        results_reg = model_reg.val(
            data=str(temp_yaml_path),
            project=str(project_root / "experiments/yolo_training/runs"),
            name="regular_eval",
            classes=relevant_coco_ids,
            verbose=False
        )
        
        print(f"Regular mAP50:       {results_reg.box.map50:.4f}")
        print(f"Regular mAP50-95:    {results_reg.box.map:.4f}")
        
        print("\n" + "="*50)
        print("Comparison Summary")
        print("="*50)
        print(f"{'Metric':<15} {'Fine-tuned':<15} {'Regular':<15} {'Diff':<15}")
        print("-" * 60)
        print(f"{'mAP50':<15} {results_ft.box.map50:<15.4f} {results_reg.box.map50:<15.4f} {results_ft.box.map50 - results_reg.box.map50:<+15.4f}")
        print(f"{'mAP50-95':<15} {results_ft.box.map:<15.4f} {results_reg.box.map:<15.4f} {results_ft.box.map - results_reg.box.map:<+15.4f}")

    except Exception as e:
        print(f"Error evaluating regular model: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary data in {temp_dir}")

if __name__ == "__main__":
    main()
