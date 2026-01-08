"""
Repickle deepfake model with the current scikit-learn version.

This script loads the existing model (default: ../models/deepfake_model.pkl),
and writes a new pickle file named `deepfake_model_sklearn_<version>.pkl`
in the same folder. It does NOT overwrite the original model.

Usage:
    python repickle_deepfake_model.py

Optional env var:
    DEEPFAKE_MODEL_PATH - path to existing model to load
"""

from __future__ import annotations

import os
import sys
import time
import traceback

try:
    import joblib
except Exception as e:
    print('joblib is required to run this script:', e)
    sys.exit(2)

try:
    import sklearn
except Exception:
    sklearn = None


DEFAULT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models', 'deepfake_model.pkl'))


def main():
    src = os.getenv('DEEPFAKE_MODEL_PATH') or DEFAULT
    print('Source model:', src)
    if not os.path.exists(src):
        print('Model file not found:', src)
        return 3

    try:
        start = time.time()
        model = joblib.load(src)
        dur = (time.time() - start) * 1000.0
        print(f'Loaded model in {dur:.0f} ms')
    except Exception as e:
        print('Failed to load model:', e)
        print(traceback.format_exc())
        return 4

    skl_ver = sklearn.__version__ if sklearn is not None else 'unknown'
    dst_name = f'deepfake_model_sklearn_{skl_ver.replace(".", "_")}.pkl'
    dst = os.path.join(os.path.dirname(src), dst_name)

    try:
        joblib.dump(model, dst)
        print('Repickled model to', dst)
        return 0
    except Exception as e:
        print('Failed to dump model:', e)
        print(traceback.format_exc())
        return 5


if __name__ == '__main__':
    sys.exit(main())
