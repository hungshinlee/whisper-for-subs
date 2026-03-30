"""
Pre-download DPDFNet TFLite models during Docker build.

Models are stored at DPDFNET_MODELS_DIR (default /app/models/dpdfnet)
so the container never needs network access at runtime.

By default downloads all four 16 kHz models.  Set DPDFNET_DOWNLOAD_MODELS
to a comma-separated list of model names to download a subset, e.g.:
    DPDFNET_DOWNLOAD_MODELS=dpdfnet4,dpdfnet8

Available names: baseline, dpdfnet2, dpdfnet4, dpdfnet8, dpdfnet2_48khz_hr
"""

import os
import sys

MODELS_DIR = os.environ.get("DPDFNET_MODELS_DIR", "/app/models/dpdfnet")
os.makedirs(MODELS_DIR, exist_ok=True)

# All available files on HuggingFace
ALL_FILES = {
    "baseline":           "baseline.tflite",
    "dpdfnet2":           "dpdfnet2.tflite",
    "dpdfnet4":           "dpdfnet4.tflite",
    "dpdfnet8":           "dpdfnet8.tflite",
    "dpdfnet2_48khz_hr":  "dpdfnet2_48khz_hr.tflite",
}

# Which models to download (default: all 16 kHz models)
_env_selection = os.environ.get("DPDFNET_DOWNLOAD_MODELS", "baseline,dpdfnet2,dpdfnet4,dpdfnet8")
selected_keys  = [k.strip() for k in _env_selection.split(",") if k.strip()]

# Validate
for key in selected_keys:
    if key not in ALL_FILES:
        print(f"⚠️  Unknown model name '{key}' — skipping. "
              f"Valid: {list(ALL_FILES)}", file=sys.stderr)

files_to_download = {k: ALL_FILES[k] for k in selected_keys if k in ALL_FILES}

if not files_to_download:
    print("ℹ️  No DPDFNet models selected for download. Skipping.")
    sys.exit(0)

print(f"📥 Downloading DPDFNet TFLite models → {MODELS_DIR}")
print(f"   Models: {list(files_to_download)}")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

for key, filename in files_to_download.items():
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.isfile(dest):
        print(f"  ✓ {filename} already exists — skipping")
        continue
    print(f"  ⬇  {filename} …")
    hf_hub_download(
        repo_id="Ceva-IP/DPDFNet",
        filename=filename,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"  ✅ {filename} saved")

print("✅ DPDFNet models ready")
