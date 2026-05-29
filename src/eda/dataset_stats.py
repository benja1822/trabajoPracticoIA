from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple
import json

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import IMAGES_DIR, ANNOTATIONS_DIR, VOC_CLASSES


def collect_image_stats(images_dir: Path = IMAGES_DIR) -> List[Dict]:
    records = []
    paths = sorted(images_dir.glob("*.jpg"))
    for p in tqdm(paths, desc="Reading image metadata", unit="img"):
        try:
            with Image.open(p) as img:
                w, h = img.size
                mode = img.mode
            size_kb = p.stat().st_size / 1024
            records.append({
                "id":      p.stem,
                "width":   w,
                "height":  h,
                "aspect":  round(w / h, 3),
                "mode":    mode,
                "size_kb": round(size_kb, 1),
            })
        except Exception:
            pass
    return records


def parse_voc_annotations(annotations_dir: Path = ANNOTATIONS_DIR) -> Dict[str, List[str]]:
    """Returns {class_name: [img_id, ...]} for all annotated images."""
    class_to_ids: Dict[str, List[str]] = defaultdict(list)
    for xml_path in tqdm(sorted(annotations_dir.glob("*.xml")), desc="Parsing XML", unit="file"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_id = root.findtext("filename", "").replace(".jpg", "")
            classes_in_image = {
                obj.findtext("name", "")
                for obj in root.findall("object")
            }
            for cls in classes_in_image:
                if cls:
                    class_to_ids[cls].append(img_id)
        except ET.ParseError:
            pass
    return dict(class_to_ids)


def class_distribution(class_to_ids: Dict[str, List[str]]) -> List[Tuple[str, int]]:
    return sorted(
        [(cls, len(ids)) for cls, ids in class_to_ids.items()],
        key=lambda x: -x[1],
    )


def print_summary(stats: List[Dict], class_to_ids: Dict[str, List[str]]) -> None:
    widths  = [r["width"]  for r in stats]
    heights = [r["height"] for r in stats]
    sizes   = [r["size_kb"] for r in stats]

    print(f"Total images      : {len(stats):,}")
    print(f"Width  — min/max/mean : {min(widths)} / {max(widths)} / {np.mean(widths):.0f} px")
    print(f"Height — min/max/mean : {min(heights)} / {max(heights)} / {np.mean(heights):.0f} px")
    print(f"Size   — min/max/mean : {min(sizes):.1f} / {max(sizes):.1f} / {np.mean(sizes):.1f} KB")
    print(f"\nAnnotated classes : {len(class_to_ids)}")
    print("\nClass distribution (top 10):")
    for cls, count in class_distribution(class_to_ids)[:10]:
        bar = "█" * (count // 100)
        print(f"  {cls:<15} {count:>5}  {bar}")
