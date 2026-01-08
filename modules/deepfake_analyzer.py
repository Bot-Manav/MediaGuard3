"""
Module: Deepfake Analyzer
Production-safe wrapper for a pretrained deepfake detection model.

Requirements and design choices:
- Load the model once at module import (cached global). Do not reload per request.
- Fail explicitly and safely if model or dependencies are missing.
- Expose a simple interface `DeepfakeAnalyzer.analyze(image)` returning a
  deterministic dict with the schema required by the product owner.
- Do NOT mix with Azure Content Safety.

The module attempts a few conservative inference strategies but will never
raise; instead it returns `analysis_failed: True` with an explanatory message.
"""

from __future__ import annotations

import os
import time
import json
import traceback
import logging
from typing import Any, Dict, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover - defensive
    Image = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover - defensive
    joblib = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - defensive
    np = None  # type: ignore


# Module logger
logger = logging.getLogger("mg.deepfake")
if not logger.handlers:
    # If no handlers configured, add a simple stream handler to ensure logs appear
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [mg.deepfake] %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


# Resolve model path relative to this file: ../models/deepfake_model.pkl
_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(_MODULE_DIR, '..', 'models', 'deepfake_model.pkl')
)


def _load_model(path: Optional[str] = None):
    # Allow explicit override via environment variable for production deployments
    env_path = os.getenv('DEEPFAKE_MODEL_PATH')
    # If DEEPFAKE_MODEL_PATH set, trust it. Otherwise, prefer a repickled
    # model that matches the runtime sklearn version: deepfake_model_sklearn_*.pkl
    if env_path:
        path = path or env_path
    else:
        # Search for repickled models in the same models directory as default
        try:
            import glob
            models_dir = os.path.dirname(_DEFAULT_MODEL_PATH)
            candidates = glob.glob(os.path.join(models_dir, 'deepfake_model_sklearn_*.pkl'))
            # Prefer highest-version candidate by lexical sort of version numbers
            if candidates:
                # Extract numeric version components for robust comparison
                def ver_key(p):
                    base = os.path.basename(p)
                    parts = base.replace('deepfake_model_sklearn_', '').replace('.pkl', '').split('_')
                    nums = []
                    for part in parts:
                        try:
                            nums.append(int(part))
                        except Exception:
                            nums.append(0)
                    return nums

                candidates.sort(key=ver_key, reverse=True)
                path = candidates[0]
            else:
                path = path or _DEFAULT_MODEL_PATH
        except Exception:
            path = path or _DEFAULT_MODEL_PATH
    if joblib is None:
        logger.error('joblib not available in runtime; deepfake model cannot be loaded')
        return None, 'joblib not available in runtime'
    if not os.path.exists(path):
        logger.warning('deepfake model file not found at %s', path)
        return None, f'model file not found: {path}'

    try:
        start = time.time()
        model = joblib.load(path)
        dur = (time.time() - start) * 1000.0
        logger.info('deepfake model loaded from %s (%.0fms)', path, dur)
        return model, None
    except Exception as e:
        logger.exception('failed to load deepfake model from %s', path)
        return None, f'failed to load model: {e}'


# Load once at import time
_MODEL, _MODEL_LOAD_ERROR = _load_model()


class DeepfakeAnalyzer:
    """Production-oriented deepfake analyzer.

    Usage:
        analyzer = DeepfakeAnalyzer()
        result = analyzer.analyze(pil_image)

    The `analyze` method never raises; it returns a dict with the following keys:
        - label: 'fake'|'real'|'unknown'
        - confidence: float 0.0-1.0
        - risk: float 0.0-1.0 (aligned with confidence for 'fake')
        - analysis_failed: bool
        - error: optional str with error details when analysis_failed is True
    """

    def __init__(self):
        self._model = _MODEL
        self._model_error = _MODEL_LOAD_ERROR
        self._model_path = os.getenv('DEEPFAKE_MODEL_PATH') or _DEFAULT_MODEL_PATH

        # Emit startup telemetry/logging
        if self._model is not None:
            logger.info('DeepfakeAnalyzer initialized; model_available=True, model_path=%s', self._model_path)
        else:
            logger.warning('DeepfakeAnalyzer initialized with no model available: %s', self._model_error)

    def analyze(self, image: Any) -> Dict[str, Any]:
        """Analyze a PIL `Image` (preferred) or return a safe failure.

        Notes:
        - Be conservative when attempting inference; catch all exceptions.
        - Normalized outputs to 0.0-1.0.
        - Do not mutate the input image.
        """
        result = {
            'label': 'unknown',
            'confidence': 0.0,
            'risk': 0.0,
            'analysis_failed': False,
            'error': None
        }

        telemetry = {
            'event': 'deepfake_inference',
            'model_path': getattr(self, '_model_path', None),
            'model_loaded': bool(self._model is not None),
            'inference_ms': None,
            'label': None,
            'confidence': None,
            'analysis_failed': None,
        }

        # Basic dependency / model checks
        if Image is None:
            result.update({'analysis_failed': True, 'error': 'PIL not available'})
            telemetry.update({'analysis_failed': True})
            logger.error('Deepfake analyze failed: PIL not available')
            _emit_telemetry_if_enabled(telemetry)
            return result

        if self._model is None:
            result.update({'analysis_failed': True, 'error': f'model unavailable: {self._model_error}'})
            telemetry.update({'analysis_failed': True})
            logger.warning('Deepfake analyze requested but model unavailable: %s', self._model_error)
            _emit_telemetry_if_enabled(telemetry)
            return result

        # Validate input type
        if not isinstance(image, Image.Image):
            result.update({'analysis_failed': True, 'error': 'input is not a PIL.Image.Image'})
            return result

        try:
            # Keep a copy; do not modify original
            img = image.copy()
            # Convert to RGB to have deterministic bytes if model expects that
            if img.mode != 'RGB':
                img = img.convert('RGB')

            start = time.time()

            # Conservative inference attempts: try common model interfaces in order.
            # 1) Pipeline-style / sklearn that accepts raw objects (some pipelines do)
            if hasattr(self._model, 'predict_proba'):
                # First try: pass the PIL image object directly (pipeline may handle it)
                try:
                    probs = self._model.predict_proba([img])
                except Exception:
                    probs = None

                # Second try: convert to numpy array (H,W,C) and pass flattened
                if probs is None and np is not None:
                    try:
                        arr = np.asarray(img)
                        sample = arr.flatten().astype(float) / 255.0
                        probs = self._model.predict_proba([sample])
                    except Exception:
                        probs = None

                if probs is not None:
                    # Binary classifier expected: class 1 -> fake
                    try:
                        fake_prob = float(probs[0][1])
                    except Exception:
                        # If shape unexpected, fallback to max-class heuristic
                        fake_prob = float(max(probs[0]))

                    fake_prob = max(0.0, min(1.0, fake_prob))
                    label = 'fake' if fake_prob >= 0.6 else ('real' if fake_prob <= 0.4 else 'unknown')
                    inference_ms = (time.time() - start) * 1000.0
                    result.update({'label': label, 'confidence': fake_prob, 'risk': fake_prob})
                    telemetry.update({'inference_ms': inference_ms, 'label': label, 'confidence': fake_prob, 'analysis_failed': False})
                    logger.info('Deepfake inference completed: label=%s confidence=%.4f ms=%.0f', label, fake_prob, inference_ms)
                    _emit_telemetry_if_enabled(telemetry)
                    return result

            # 2) Models with `predict` only
            if hasattr(self._model, 'predict'):
                try:
                    pred = self._model.predict([img])
                except Exception:
                    pred = None

                if pred is None and np is not None:
                    try:
                        arr = np.asarray(img)
                        sample = arr.flatten().astype(float) / 255.0
                        pred = self._model.predict([sample])
                    except Exception:
                        pred = None

                if pred is not None:
                    # Heuristic: if prediction is probabilistic-like
                    try:
                        p = float(pred[0])
                        fake_prob = max(0.0, min(1.0, p))
                    except Exception:
                        # If classes returned as labels, map common labels
                        label_val = str(pred[0]).lower()
                        if label_val in ('fake', '1', 'true', 'yes'):
                            fake_prob = 0.95
                        elif label_val in ('real', '0', 'false', 'no'):
                            fake_prob = 0.05
                        else:
                            fake_prob = 0.5

                    inference_ms = (time.time() - start) * 1000.0
                    label = 'fake' if fake_prob >= 0.6 else ('real' if fake_prob <= 0.4 else 'unknown')
                    result.update({'label': label, 'confidence': fake_prob, 'risk': fake_prob})
                    telemetry.update({'inference_ms': inference_ms, 'label': label, 'confidence': fake_prob, 'analysis_failed': False})
                    logger.info('Deepfake inference completed: label=%s confidence=%.4f ms=%.0f', label, fake_prob, inference_ms)
                    _emit_telemetry_if_enabled(telemetry)
                    return result

            # If none of the above worked, we can't run the model safely
            inference_ms = (time.time() - start) * 1000.0
            result.update({'analysis_failed': True, 'error': 'model does not support a known image inference API'})
            telemetry.update({'inference_ms': inference_ms, 'analysis_failed': True})
            logger.error('Deepfake inference failed: unsupported model API (inference_ms=%.0f)', inference_ms)
            _emit_telemetry_if_enabled(telemetry)
            return result

        except Exception as exc:  # pragma: no cover - defensive
            tb = traceback.format_exc()
            result.update({'analysis_failed': True, 'error': f'analysis exception: {exc}\n{tb}'})
            telemetry.update({'analysis_failed': True, 'error': str(exc)})
            logger.exception('Deepfake analysis exception')
            _emit_telemetry_if_enabled(telemetry)
            return result


__all__ = ['DeepfakeAnalyzer']


def _emit_telemetry_if_enabled(payload: Dict[str, Any]):
    """Emit structured telemetry if `DEEPFAKE_TELEMETRY=1` is set.

    The emission method is intentionally minimal: write a JSON line to
    the logger at INFO level. This keeps the dependency surface small and
    allows platform log collectors to ingest the events.
    """
    try:
        if os.getenv('DEEPFAKE_TELEMETRY', '0') != '1':
            return
        # Ensure values are JSON-serializable
        j = json.dumps(payload, default=str)
        logger.info('TELEMETRY %s', j)
    except Exception:
        # Telemetry must never break inference.
        logger.exception('Failed to emit deepfake telemetry')
