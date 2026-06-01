from pathlib import Path
from typing import List, Optional, Tuple
import json

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

from src.config import (
    CLIP_MODEL_NAME, CLIP_BATCH_SIZE, DEVICE, SEED,
    EMBEDDINGS_FILE, IMAGE_IDS_FILE, IMAGES_DIR,
)


class CLIPExtractor:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._model_name = model_name

    def _load(self) -> None:
        if self._model is None:
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            self._model = CLIPModel.from_pretrained(self._model_name).to(self.device)
            self._model.eval()

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[Path],
        batch_size: int = CLIP_BATCH_SIZE,
        normalize: bool = True,
    ) -> np.ndarray:
        self._load()
        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images", unit="batch"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))

            inputs = self._processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._model.get_image_features(**inputs)

            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)

            all_embeddings.append(feats.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        self._load()
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self._model.get_text_features(**inputs)

        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode_texts([text], normalize=normalize)[0]


def build_and_save_embeddings(
    images_dir: Path = IMAGES_DIR,
    embeddings_file: Path = EMBEDDINGS_FILE,
    ids_file: Path = IMAGE_IDS_FILE,
    batch_size: int = CLIP_BATCH_SIZE,
) -> Tuple[np.ndarray, List[str]]:
    image_paths = sorted(images_dir.glob("*.jpg"))
    image_ids   = [p.stem for p in image_paths]

    extractor = CLIPExtractor()
    embeddings = extractor.encode_images(image_paths, batch_size=batch_size)

    np.save(embeddings_file, embeddings)
    ids_file.write_text(json.dumps(image_ids), encoding="utf-8")

    print(f"Saved {len(image_ids)} embeddings → {embeddings_file}")
    return embeddings, image_ids


def load_embeddings(
    embeddings_file: Path = EMBEDDINGS_FILE,
    ids_file: Path = IMAGE_IDS_FILE,
) -> Tuple[np.ndarray, List[str]]:
    embeddings = np.load(embeddings_file)
    image_ids  = json.loads(ids_file.read_text(encoding="utf-8"))
    return embeddings, image_ids
