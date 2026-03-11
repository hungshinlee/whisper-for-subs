"""Pre-download DeepFilterNet3 model during Docker build."""
import sys
import types

import torchaudio

# torchaudio 2.1+ removed torchaudio.backend.common; patch it so deepfilternet imports cleanly
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

from df import init_df  # noqa: E402

init_df("DeepFilterNet3", log_level="none")
print("DeepFilterNet3 ready")
