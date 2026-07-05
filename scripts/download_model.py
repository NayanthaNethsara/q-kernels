import os
import sys

from huggingface_hub import snapshot_download

# Usage: python scripts/download_model.py [modelName]
# Downloads the raw model files into models/<modelName> so other scripts
# can load them from disk instead of re-downloading from the Hub.
modelName = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
huggingFaceToken = os.environ.get("HF_TOKEN", None)

repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
localModelDir = os.path.join(repoRoot, "models", modelName)

downloadedPath = snapshot_download(
    repo_id=modelName,
    local_dir=localModelDir,
    token=huggingFaceToken,
)

print(f"Model '{modelName}' saved to: {downloadedPath}")
