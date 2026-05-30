"""Build and persist ground truth sets for evaluation.

q1–q20 : derived from official Pascal VOC 2012 XML annotations (20 classes).
q21–q40: loaded from the cátedra's solution.csv, or approximated via CLIP
         intersection for local development.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import numpy as np
from tqdm import tqdm

from src.config import (
    ANNOTATIONS_DIR, VOC_CLASSES, VOC_GT_FILE, COMPLEX_GT_FILE,
)

# Official mapping: query id → VOC class name (q1 = first class, etc.)
VOC_QUERY_MAP: Dict[str, str] = {
    f"q{i+1}": cls for i, cls in enumerate(VOC_CLASSES)
}


def build_voc_ground_truth(
    annotations_dir: Path = ANNOTATIONS_DIR,
    save_path: Optional[Path] = VOC_GT_FILE,
) -> Dict[str, Set[str]]:
    """Parse all XML annotations and return {class_name: {img_id, ...}}."""
    from collections import defaultdict
    class_to_ids: Dict[str, Set[str]] = defaultdict(set)

    for xml_path in tqdm(sorted(annotations_dir.glob("*.xml")),
                         desc="Building VOC ground truth", unit="file"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_id = root.findtext("filename", "").replace(".jpg", "")
            for obj in root.findall("object"):
                cls = obj.findtext("name", "")
                if cls:
                    class_to_ids[cls].add(img_id)
        except ET.ParseError:
            pass

    result = {cls: ids for cls, ids in class_to_ids.items() if cls in VOC_CLASSES}

    if save_path:
        serializable = {k: sorted(v) for k, v in result.items()}
        save_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    return result


def load_voc_ground_truth(path: Path = VOC_GT_FILE) -> Dict[str, Set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: set(v) for k, v in raw.items()}


def build_query_gt_from_voc(
    voc_gt: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Map q1–q20 to their corresponding VOC class ground truth."""
    return {
        qid: voc_gt.get(cls, set())
        for qid, cls in VOC_QUERY_MAP.items()
    }


def load_solution_csv(solution_csv_path: Path) -> Dict[str, Set[str]]:
    """Load the cátedra's solution.csv as ground truth for q21–q40.

    Expected format: qid,preds  where preds is semicolon-separated image IDs.
    """
    import csv
    gt: Dict[str, Set[str]] = {}
    with open(solution_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid   = row["qid"].strip()
            ids   = {iid.strip() for iid in row["preds"].split(";") if iid.strip()}
            gt[qid] = ids
    return gt


def approximate_complex_gt(
    query_text: str,
    embeddings: np.ndarray,
    image_ids: List[str],
    clip_extractor,
    top_n: int = 200,
) -> Set[str]:
    """Approximate ground truth for a complex query via CLIP top-N retrieval.

    Used only for local ablation — the official GT comes from the cátedra.
    """
    query_emb = clip_extractor.encode_text(query_text)
    scores    = embeddings @ query_emb
    top_idx   = np.argsort(-scores)[:top_n]
    return {image_ids[i] for i in top_idx}
