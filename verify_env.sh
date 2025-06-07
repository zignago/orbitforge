#!/usr/bin/env bash
#
# orbitforge_env_check.sh
# Quick health-check for OrbitForge development environment on WSL-Ubuntu.
#
# Usage:
#   chmod +x orbitforge_env_check.sh
#   ./orbitforge_env_check.sh
#
# Exit codes
#   0  – all mandatory checks pass
#   1  – at least one mandatory check fails
#   2  – only warnings (non-fatal)

set -euo pipefail

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

declare -a errors=()
declare -a warnings=()
declare -A fixes=()

note()  { echo -e "${BLUE}$*${RESET}"; }
ok()    { echo -e "  ${GREEN}✔${RESET} $*"; }
fail()  { echo -e "  ${RED}✖${RESET} $1"; errors+=("$1"); fixes["$1"]="$2"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET} $1"; warnings+=("$1"); fixes["$1"]="$2"; }

note "OrbitForge environment smoke-test — $(date)"
echo

###############################################################################
# 0.  Operating system
###############################################################################
note "0. Operating system"
os_name=$(lsb_release -ds 2>/dev/null || echo "Unknown")
kernel=$(uname -r)
echo "  Detected: $os_name, kernel $kernel"

[[ $os_name == *"Ubuntu 22.04"* ]] \
  && ok "Ubuntu 22.04" \
  || warn "Ubuntu 22.04 LTS not detected" \
          "Install Ubuntu-22.04 via 'wsl --install -d Ubuntu-22.04'."

[[ $kernel == *"microsoft-standard"* ]] \
  && ok "WSL 2 kernel" \
  || warn "Kernel does not look like WSL 2" \
          "Run 'wsl --set-version <distro> 2' then restart WSL."

###############################################################################
# 1.  GPU & CUDA
###############################################################################
note "1. GPU & CUDA"

if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
           | head -n1 | tr -d ' ')
    echo "  Detected GPU: $gpu_name, ${vram} MiB VRAM"

    ok "nvidia-smi present"
    if (( vram < 10000 )); then
        warn "GPU VRAM < 10 GB (detected ${vram} MiB)" \
             "Large 128³ voxel batches may OOM; lower batch size or enable gradient checkpointing."
    else
        ok "Adequate VRAM: ${vram} MiB"
    fi
else
    fail "nvidia-smi not installed / GPU not exposed in WSL" \
         "Install NVIDIA driver ≥ 535 in Windows, enable WSL 2 GPU, then reboot."
fi

if command -v nvcc &>/dev/null; then
    cuda_ver=$(nvcc --version | grep release | grep -o "[0-9]\+\.[0-9]\+" | head -n1)
    ok "nvcc version $cuda_ver"
    (( ${cuda_ver%%.*} < 12 )) \
        && warn "CUDA < 12 detected" \
                "Install 'cuda-toolkit-12-3' and reinstall PyTorch with CUDA 12 support."
else
    fail "nvcc not found" "Install 'cuda-toolkit-12-3' inside WSL."
fi

###############################################################################
# 2.  Conda & PyTorch
###############################################################################
note "2. Conda environment"

if command -v conda &>/dev/null; then
    env_list=$(conda env list)
    if echo "$env_list" | grep -q "^orbitforge"; then
        current=$(echo "$env_list" | awk '/\*/ {print $1}')
        [[ $current == "orbitforge" ]] && ok "orbitforge env active" \
            || warn "'orbitforge' exists but not active" "Run 'conda activate orbitforge'."
    else
        fail "conda env 'orbitforge' missing" \
             "Create via 'mamba create -n orbitforge python=3.11' then reinstall deps."
    fi
else
    fail "conda/mamba not installed" \
         "Install Mambaforge: https://github.com/conda-forge/miniforge/releases"
fi

# PyTorch CUDA check (needs jq)
if ! command -v jq &>/dev/null; then
    warn "jq not installed (needed for JSON parsing)" \
         "sudo apt-get install -y jq"
fi

python - <<'PY' > /tmp/torch_check.json || echo "__PYFAIL__" > /tmp/torch_check.json
import json, torch, sys
json.dump({"available": torch.cuda.is_available(),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
           "torch": torch.__version__}, sys.stdout)
PY

if grep -q "__PYFAIL__" /tmp/torch_check.json; then
    fail "Python/PyTorch failed to run" \
         "Ensure conda env has PyTorch built with CUDA 12 support."
else
    torch_available=$(jq -r '.available' /tmp/torch_check.json)
    torch_device=$(jq -r '.device'    /tmp/torch_check.json)
    torch_ver=$(jq -r '.torch'        /tmp/torch_check.json)
    ok "PyTorch ${torch_ver} — CUDA available: ${torch_available} on ${torch_device}"
    [[ $torch_available != "true" ]] \
        && fail "PyTorch cannot see CUDA" \
               "Install 'pytorch-cuda' that matches your CUDA toolkit."
fi

###############################################################################
# 3.  Git toolchain
###############################################################################
note "3. Git, LFS, DVC"

for bin in git git-lfs dvc; do
    if command -v $bin &>/dev/null; then
        ok "$($bin --version | head -n1)"
    else
        fail "$bin not installed" "Install via apt or mamba: 'sudo apt-get install $bin'"
    fi
done

###############################################################################
# 4.  Docker GPU
###############################################################################
note "4. Docker & GPU passthrough"

if command -v docker &>/dev/null; then
    ok "$(docker --version)"
    if docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi -L \
        >/dev/null 2>&1; then
        ok "Docker can access GPU"
    else
        warn "Docker cannot access GPU" \
             "Enable WSL integration & GPU support in Docker Desktop → Settings."
    fi
else
    fail "docker not installed" \
         "Install Docker Desktop on Windows and enable WSL backend."
fi

###############################################################################
# 5.  Node (optional frontend work)
###############################################################################
note "5. Node & npm"

if command -v node &>/dev/null; then
    node_ver=$(node -v)
    ok "Node ${node_ver}"
    [[ ${node_ver#v} < 18 ]] \
        && warn "Node < 18 detected" "Install Node 18 LTS via nvm."
else
    warn "Node not installed" \
         "Install Node 18 via nvm if you plan to develop the React frontend."
fi

###############################################################################
# 6.  System resources snapshot
###############################################################################
note "6. System resources"
total_mem=$(free -g | awk '/Mem:/ {print $2}')
(( total_mem < 16 )) \
    && warn "System RAM < 16 GB (detected ${total_mem} GB)" \
            "Large training runs may swap; close apps or upgrade RAM." \
    || ok "RAM ${total_mem} GB"

free_disk=$(df -h ~ | awk 'NR==2 {print $4}')
echo "  Free disk in home: ${free_disk}"

###############################################################################
# 7.  Summary & exit code
###############################################################################
echo
if (( ${#errors[@]} )); then
    echo -e "${RED}✖ ${#errors[@]} mandatory checks FAILED:${RESET}"
    for e in "${errors[@]}"; do
        echo "   • $e"
        echo "     → ${fixes[$e]}"
    done
    exit 1
else
    echo -e "${GREEN}✓ All mandatory checks passed.${RESET}"
    if (( ${#warnings[@]} )); then
        echo -e "${YELLOW}⚠ ${#warnings[@]} warnings:${RESET}"
        for w in "${warnings[@]}"; do
            echo "   • $w"
            echo "     → ${fixes[$w]}"
        done
        exit 2
    fi
    exit 0
fi
