#!/bin/bash
set -uo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/SpectraBreast-Vision/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${HOME}/SpectraBreast-Vision/mast3r/dust3r/croco/models:${PYTHONPATH:-}"
cd ~/SpectraBreast-Vision
PASS=0; FAIL=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  [OK]   $1"; PASS=$((PASS+1)); else echo "  [FAIL] $1"; FAIL=$((FAIL+1)); fi; }
echo "=== PREFLIGHT CHECK ($(hostname)) ==="
chk "uv"                  "command -v uv"
chk ".venv"               "[ -d .venv ]"
chk "config"              "[ -f configs/default.yaml ]"
chk "GPU"                 "nvidia-smi"
chk "torch"               "uv run python -c "import torch""
chk "CUDA"                "uv run python -c "import torch; assert torch.cuda.is_available()""
chk "curope.rope_2d"      "uv run python -c "import torch,curope; assert hasattr(curope,'rope_2d')""
chk "spectra"             "uv run python -c "import spectra""
chk "mast3r"              "uv run python -c "import mast3r""
chk "DATA dir"            "[ -d DATA ]"
chk "rope_2d su GPU"      "uv run python -c "import torch,curope; t=torch.randn(1,4,10,8,device='cuda'); p=torch.zeros(1,10,2,dtype=torch.long,device='cuda'); curope.rope_2d(t,p,100.0,1.0); torch.cuda.synchronize()""
echo "=== RISULTATO: ${PASS} OK, ${FAIL} FAIL ==="
