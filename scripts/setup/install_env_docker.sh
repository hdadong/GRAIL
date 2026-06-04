#!/usr/bin/env bash
# GRAIL Docker setup — installs Blender into the bind-mounted repo.
# The Docker image already has conda envs, Python deps, and CUDA extensions.
# Usage: bash scripts/setup/install_env_docker.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

# Init conda
CONDA_BIN=""
for p in /root/miniconda3 /opt/conda /root/miniforge3; do
    [ -f "$p/bin/conda" ] && CONDA_BIN="$p/bin/conda" && break
done
[ -z "$CONDA_BIN" ] && echo -e "${RED}conda not found. Are you in the GRAIL Docker container?${NC}" && exit 1
eval "$($CONDA_BIN shell.bash hook)"
$CONDA_BIN init bash 2>/dev/null | tail -1

# Verify grail env
conda env list 2>/dev/null | grep -q "^grail " || { echo -e "${RED}'grail' env not found. Use install_env_scratch.sh instead.${NC}"; exit 1; }

# Pin setuptools<81 in hunyuan env: lightning_fabric (transitive dep via
# pytorch_lightning) calls pkg_resources.declare_namespace(), which setuptools
# 81 removed. The pre-built Docker image ships setuptools 82+.
if conda env list 2>/dev/null | grep -q "^hunyuan "; then
    conda run -n hunyuan pip install --quiet 'setuptools<81' 2>&1 | tail -1

    # Build mesh_inpaint_processor.so if missing: the prebuilt image is missing
    # this extension, which causes meshVerticeInpaint NameError in texture inpaint.
    DR_DIR="$PROJECT_ROOT/imports/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer"
    if [ -f "$DR_DIR/compile_mesh_painter.sh" ] && ! ls "$DR_DIR"/mesh_inpaint_processor*.so &>/dev/null; then
        echo "Building mesh_inpaint_processor.so..."
        (cd "$DR_DIR" && conda run -n hunyuan --no-capture-output bash compile_mesh_painter.sh) 2>&1 | tail -3
    fi
fi

# Build FoundationPose mycpp.so if missing: prebuilt image ships source only,
# and bundled build_all_conda.sh doesn't pass pybind11_DIR.
FP_MYCPP_DIR="$PROJECT_ROOT/imports/FoundationPose/mycpp"
if [ -f "$FP_MYCPP_DIR/CMakeLists.txt" ] && ! ls "$FP_MYCPP_DIR"/build/mycpp*.so &>/dev/null; then
    echo "Building FoundationPose mycpp.so..."
    PYBIND11_DIR=$(conda run -n grail python -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)
    (cd "$FP_MYCPP_DIR" && rm -rf build && mkdir -p build && cd build && \
        cmake -Dpybind11_DIR="$PYBIND11_DIR" .. && make -j"$(nproc)") 2>&1 | tail -3
fi

# Download Blender
BLENDER_DIR="$PROJECT_ROOT/imports/blender"
if [ ! -x "$BLENDER_DIR/blender" ]; then
    echo "Downloading Blender 4.3.2..."
    mkdir -p "$BLENDER_DIR"
    curl -L "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" -o /tmp/blender.tar.xz
    tar -xf /tmp/blender.tar.xz -C "$BLENDER_DIR" --strip-components=1
    rm -f /tmp/blender.tar.xz
    echo -e "${GREEN}Blender installed.${NC}"
else
    echo "Blender already installed."
fi

# Blender Python deps — numpy pinned to match grail env so pickles exchanged
# between the two (e.g. initial_state.pickle) deserialize cleanly. Run on every
# invocation so an existing Blender install also gets the right numpy.
BLENDER_PY="$BLENDER_DIR/4.3/python/bin/python3.11"
if [ -x "$BLENDER_PY" ]; then
    $BLENDER_PY -m ensurepip 2>&1 | tail -1
    $BLENDER_PY -m pip install 'numpy==2.2.6' pyyaml trimesh Pillow imageio scipy tqdm omegaconf opencv-python 2>&1 | tail -3
    $BLENDER_PY -m pip install -e "$PROJECT_ROOT" --no-deps 2>&1 | tail -1
fi

echo -e "${GREEN}Docker setup complete.${NC}"
