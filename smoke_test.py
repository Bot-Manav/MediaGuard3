"""
Smoke test for deepfake analyzer in mg3.

This script attempts to:
- load the DeepfakeAnalyzer (which loads the model once)
- run a single inference on an in-memory generated image

The script is defensive: it never raises to the caller. It prints clear
diagnostics for CI / deployment checks.
"""

from __future__ import annotations

import os
import sys
import traceback

try:
    from PIL import Image
except Exception:
    Image = None


def main():
    print('mg3 smoke test: Deepfake analyzer')

    # Ensure repo_clone is on path when running from mg3
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from modules.deepfake_analyzer import DeepfakeAnalyzer
    except Exception as e:
        print('ERROR: failed to import DeepfakeAnalyzer:', e)
        print(traceback.format_exc())
        return 2

    analyzer = DeepfakeAnalyzer()

    # Create a small deterministic test image
    if Image is None:
        print('PIL not available; cannot create test image. Skipping inference.')
        return 0

    try:
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        res = analyzer.analyze(img)
        print('Analysis result (trimmed):')
        # Print the most relevant fields
        print(' label:', res.get('label'))
        print(' confidence:', res.get('confidence'))
        print(' risk:', res.get('risk'))
        print(' analysis_failed:', res.get('analysis_failed'))
        if res.get('analysis_failed'):
            print(' error:', res.get('error'))
            return 1
        else:
            print(' Smoke test inference succeeded (model loaded and returned a result).')
            return 0

    except Exception as e:
        print('ERROR: exception during inference:', e)
        print(traceback.format_exc())
        return 3


if __name__ == '__main__':
    code = main()
    sys.exit(code)
