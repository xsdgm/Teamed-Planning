#!/usr/bin/env bash
# download_models.sh
# Downloads SegFormer-B0 ONNX weights from HuggingFace into this directory.
# Run once on the host before starting the system:
#
#   bash src/teamed-planning/scripts/drone/models/download_models.sh
#
# The models/ directory is bind-mounted into the planning container at:
#   /home/dcist/Teamed-Planning/src/teamed-planning/scripts/drone/models/
# so no docker cp is needed after downloading.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/segformer_b0_ade"
MODEL_FILE="${MODEL_DIR}/model.onnx"

echo "[download_models] Target: ${MODEL_FILE}"

if [ -f "${MODEL_FILE}" ] && [ -s "${MODEL_FILE}" ]; then
    echo "[download_models] Already exists ($(du -sh "${MODEL_FILE}" | cut -f1)) — skipping download."
    exit 0
elif [ -f "${MODEL_FILE}" ]; then
    echo "[download_models] Found empty/incomplete file, re-downloading..."
    rm -f "${MODEL_FILE}"
fi

mkdir -p "${MODEL_DIR}"

# nvidia/segformer-b0-finetuned-ade-512-512 has no pre-built model.onnx.
# We download the PyTorch weights and export to ONNX using torch.onnx.export.
# Uses hf-mirror.com (accessible without VPN) as the download source.
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="${MIRROR}/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main"
TMP_DIR="$(mktemp -d)"

echo "[download_models] Downloading PyTorch weights from ${MIRROR}..."
wget -q --show-progress -O "${TMP_DIR}/config.json"        "${REPO}/config.json"
wget -q --show-progress -O "${TMP_DIR}/pytorch_model.bin"  "${REPO}/pytorch_model.bin"

echo "[download_models] Exporting to ONNX (requires torch + transformers)..."
python3 - <<PYEOF
from transformers import SegformerForSemanticSegmentation
import torch, os, shutil, sys

tmp = "${TMP_DIR}"
out = "${MODEL_FILE}"

try:
    model = SegformerForSemanticSegmentation.from_pretrained(tmp)
    model.eval()
    dummy = torch.zeros(1, 3, 512, 512)
    torch.onnx.export(
        model, dummy, out, opset_version=11,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}})
    print(f"[download_models] ONNX saved: {os.path.getsize(out)/1e6:.1f} MB")
except Exception as e:
    print(f"[download_models] Export failed: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PYEOF

echo "[download_models] Done. Model size: $(du -sh "${MODEL_FILE}" | cut -f1)"
