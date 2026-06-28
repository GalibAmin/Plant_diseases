# Models

This folder is where the dashboard (`dashboard/app.py`) expects to find the trained `.keras` checkpoints. They are **intentionally excluded from git** (see root `.gitignore`) because GitHub blocks pushes >100MB and discourages anything over 50MB without Git LFS — and ResNet50/VGG16 checkpoints easily exceed that.

## Expected files

| Filename | Model | Approx. size |
|---|---|---|
| `mobilenet_best.keras` | MobileNetV2 | ~12.5 MB |
| `20epochvgg16model.keras` | VGG16 | Large (100MB+) |
| `100epochrenet.keras` | ResNet50 | Large (100MB+) |
| `30eppochefficientnet.keras` | EfficientNetB0 | Medium |

You only need **at least one** of these for the dashboard to run — it skips whatever's missing.

## How to get them onto this machine

Pick whichever fits your workflow:

### Option A — Git LFS (simplest if you want everything in one repo)
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes models/*.keras
git commit -m "Add trained model weights via Git LFS"
git push
```

### Option B — Hugging Face Hub (recommended for portfolio visibility)
1. Create a model repo at [huggingface.co/new](https://huggingface.co/new).
2. Upload your `.keras` files there.
3. In `dashboard/app.py`, download them at runtime instead of expecting a local file:
   ```python
   from huggingface_hub import hf_hub_download
   path = hf_hub_download(repo_id="<your-username>/plant-disease-models", filename="mobilenet_best.keras")
   ```

### Option C — Google Drive (fastest, least "professional" for a public repo)
Host the files on Drive, then add a small download step using `gdown`:
```bash
pip install gdown
gdown --id <FILE_ID> -O models/mobilenet_best.keras
```

## Regenerating from scratch

If you'd rather retrain than download, run `notebooks/01_kaggle_training.ipynb` on Kaggle (free GPU), then copy the saved `.keras` files from `/kaggle/working/` into this folder.
