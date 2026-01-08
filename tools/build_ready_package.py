"""
Build a self-contained MediaGuard_ready folder suitable for upload.

This script copies selected files from the current workspace into a
single folder outside the repo (by default: ../MediaGuard_ready) and
excludes temporary files.

Run from repository root (repo_clone) with:
    python tools/build_ready_package.py

It will create ../MediaGuard_ready with the following structure:
- streamlit_app.py
- requirements.txt
- .env.example
- DEPLOYMENT_CHECKLIST.md
- modules/ (all source .py files)
- models/ (deepfake_model_sklearn_*.pkl and optional fake-news models)
- tools/repickle_deepfake_model.py
- mediaguard_datasets/ (selected CSVs)
- .github/workflows/deploy.yml
- README.md

This script is idempotent and will overwrite the target folder.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent / 'MediaGuard_ready'

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def copy_tree(src: Path, dst: Path, ignore=None):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)

def ignore_filter(dir, contents):
    # Exclude __pycache__, .streamlit, .DS_Store
    return {c for c in contents if c in ('__pycache__', '.streamlit') or c.endswith('.DS_Store')}

def main():
    print('Building MediaGuard_ready at', TARGET)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()

    # Top-level files
    files_to_copy = [
        ROOT / 'streamlit_app.py',
        ROOT / 'requirements.txt',
        ROOT / 'mg3' / 'DEPLOYMENT_CHECKLIST.md',
        ROOT / 'mg3' / 'smoke_test.py',
    ]

    for f in files_to_copy:
        if f.exists():
            copy_file(f, TARGET / f.name)

    # Copy .env.example from workspace root if present
    workspace_env = ROOT.parent / '.env.example'
    if workspace_env.exists():
        copy_file(workspace_env, TARGET / '.env.example')

    # Copy modules directory (source code)
    src_modules = ROOT / 'modules'
    if src_modules.exists():
        copy_tree(src_modules, TARGET / 'modules', ignore=ignore_filter)

    # Copy models: include deepfake repickled model and fake-news models if present
    src_models = ROOT / 'models'
    dst_models = TARGET / 'models'
    ensure_dir(dst_models)
    if src_models.exists():
        for p in src_models.glob('*'):
            if p.is_file() and (p.name.startswith('deepfake_model') or p.name.startswith('fake_news')):
                shutil.copy2(p, dst_models / p.name)

    # Copy tools: repickle script
    src_tools = ROOT / 'tools'
    if src_tools.exists():
        ensure_dir(TARGET / 'tools')
        for p in src_tools.glob('*.py'):
            shutil.copy2(p, TARGET / 'tools' / p.name)

    # Copy mediaguard datasets CSVs (FakeNewsNet) from workspace root
    src_datasets = ROOT.parent / 'mediaguard_datasets' / 'repos' / 'FakeNewsNet' / 'dataset'
    if src_datasets.exists():
        dst_ds = TARGET / 'mediaguard_datasets' / 'repos' / 'FakeNewsNet' / 'dataset'
        dst_ds.mkdir(parents=True, exist_ok=True)
        for p in src_datasets.glob('*.csv'):
            shutil.copy2(p, dst_ds / p.name)

    # Write README.md
    readme = TARGET / 'README.md'
    readme.write_text("""
# MediaGuard_ready

This package contains a ready-to-upload copy of MediaGuard with deepfake
and fake-news integrations. To run locally after cloning, follow these steps:

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. (Optional) Set environment variables in `.env` or in your environment. See `.env.example`.

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py --server.port 8502 --server.address 0.0.0.0
```

4. (Optional) Run smoke test to verify deepfake model loads:

```powershell
python mg3/smoke_test.py
```

Deployment
- See `DEPLOYMENT_CHECKLIST.md` for Azure App Service deployment guidance.
""")

    # Create .github workflow
    gh_dir = TARGET / '.github' / 'workflows'
    gh_dir.mkdir(parents=True, exist_ok=True)
    workflow = gh_dir / 'deploy.yml'
    workflow.write_text(_workflow_content())

    print('MediaGuard_ready built at', TARGET)

def _workflow_content():
    return '''name: Deploy MediaGuard

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run smoke test (optional)
      env:
        DEEPFAKE_MODEL_PATH: ${{ secrets.DEEPFAKE_MODEL_PATH }}
      run: |
        python mg3/smoke_test.py || true

    - name: 'Login to Azure'
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: 'Deploy to Azure WebApp'
      uses: azure/webapps-deploy@v2
      with:
        app-name: ${{ secrets.AZURE_WEBAPP_NAME }}
        package: '.'
'''

if __name__ == '__main__':
    main()
