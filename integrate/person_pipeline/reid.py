"""Re-ID module: person crop -> L2-normalized appearance embedding (ONNX Runtime).

Default model: NVIDIA TAO ReIdentificationNet (ResNet50, Market-1501), input 1x3x256x128 RGB,
preprocessing /255 then per-channel mean/std, output a 256-d feature.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .config import ReIDConfig


class ReIDExtractor:
    def __init__(self, cfg: ReIDConfig):
        self.cfg = cfg
        available = ort.get_available_providers()
        providers = [p for p in cfg.providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(cfg.model, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Export may have a fixed batch of 1; batch all crops in one call only when the batch axis is dynamic.
        self.dynamic_batch = not isinstance(inp.shape[0], int)
        self.width, self.height = cfg.input_size
        self.mean = np.asarray(cfg.mean, np.float32)
        self.std = np.asarray(cfg.std, np.float32)
        self.dim = self.session.get_outputs()[0].shape[-1]
        print(f"[reid] {cfg.model} on {self.session.get_providers()[0]}, batch={'dynamic' if self.dynamic_batch else 1}")

    def _preprocess(self, image: np.ndarray, box) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), w), min(int(y2), h)
        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        x = (crop.astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)  # HWC -> CHW

    def extract(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(K, 4) xyxy boxes -> (K, D) L2-normalized embeddings."""
        if len(boxes) == 0:
            return np.zeros((0, self.dim), np.float32)
        batch = np.stack([self._preprocess(image, b) for b in boxes])
        if self.dynamic_batch:
            feats = self.session.run(None, {self.input_name: batch})[0]
        else:
            feats = np.concatenate([self.session.run(None, {self.input_name: x[None]})[0] for x in batch])
        feats = feats.reshape(len(boxes), -1).astype(np.float32)
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
