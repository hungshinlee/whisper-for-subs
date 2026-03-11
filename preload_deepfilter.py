"""
Pre-download DeepFilterNet3 model during Docker build.

The model is stored at DF_PRETRAINED_MODELS_PATH (set to /app/models in the
Dockerfile), so the container never needs network access at runtime.
"""
import os
import sys
import types

import torchaudio

# ── torchaudio compat shim ────────────────────────────────────────────────────
# deepfilternet <=0.5.6 imports torchaudio.backend.common.AudioMetaData, which
# was removed in torchaudio 2.1+.  Inject a stub so the import succeeds.
try:
    from torchaudio.backend.common import AudioMetaData  # noqa: F401
except (ImportError, ModuleNotFoundError):
    AM = getattr(torchaudio, "AudioMetaData", type("AudioMetaData", (), {}))
    bm = types.ModuleType("torchaudio.backend")
    cm = types.ModuleType("torchaudio.backend.common")
    cm.AudioMetaData = AM
    bm.common = cm
    sys.modules.setdefault("torchaudio.backend", bm)
    sys.modules["torchaudio.backend.common"] = cm

# ── Download / cache model ────────────────────────────────────────────────────
from df import init_df  # noqa: E402

model_dir = os.environ.get("DF_PRETRAINED_MODELS_PATH", "/app/models")
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading DeepFilterNet3 → {model_dir} …")
init_df("DeepFilterNet3", log_level="none")
print("DeepFilterNet3 ready")
