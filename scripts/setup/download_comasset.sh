#!/usr/bin/env bash
# ============================================================================
# Download the ComAsset dataset from HuggingFace.
#
# Usage:
#   bash scripts/setup/download_comasset.sh                    # download all categories
#   bash scripts/setup/download_comasset.sh --category axe     # download specific category
#
# Source: https://huggingface.co/datasets/SShowbiz/ComAsset
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/ComAsset"

CATEGORY=""

for arg in "$@"; do
    case "$arg" in
        --category=*) CATEGORY="${arg#*=}" ;;
        --category)   shift_next=true ;;
        --help|-h)
            echo "Usage: bash scripts/setup/download_comasset.sh [--category <name>]"
            exit 0
            ;;
        *)
            if [ "${shift_next:-false}" = true ]; then
                CATEGORY="$arg"
                shift_next=false
            fi
            ;;
    esac
done

echo "Downloading ComAsset dataset to $DATA_DIR..."
mkdir -p "$DATA_DIR"

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download SShowbiz/ComAsset \
        --repo-type dataset \
        --local-dir "$DATA_DIR" \
        --local-dir-use-symlinks False
else
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='SShowbiz/ComAsset',
    repo_type='dataset',
    local_dir='$DATA_DIR',
    local_dir_use_symlinks=False,
)
print('Download complete.')
"
fi

# HuggingFace repo nests categories under a `data/` subdirectory; grail code
# expects them at the top level. Flatten after download.
if [ -d "$DATA_DIR/data" ]; then
    echo "Flattening $DATA_DIR/data/* -> $DATA_DIR/ ..."
    shopt -s dotglob nullglob
    for entry in "$DATA_DIR/data"/*; do
        name=$(basename "$entry")
        if [ -e "$DATA_DIR/$name" ]; then
            echo "  skip (already exists at top level): $name"
        else
            mv "$entry" "$DATA_DIR/"
        fi
    done
    shopt -u dotglob nullglob
    rmdir "$DATA_DIR/data" 2>/dev/null || true
fi

# Categories ship with spaces ("cordless drill"); grail code / configs expect
# underscores ("cordless_drill"). Normalize directory names.
echo "Normalizing category names (spaces -> underscores) ..."
renamed=0
for entry in "$DATA_DIR"/*/; do
    name=$(basename "$entry")
    if [[ "$name" == *" "* ]]; then
        new="${name// /_}"
        if [ -e "$DATA_DIR/$new" ]; then
            echo "  skip (target exists): $name -> $new"
        else
            mv -- "$DATA_DIR/$name" "$DATA_DIR/$new"
            renamed=$((renamed+1))
        fi
    fi
done
echo "  renamed $renamed categor$( [ "$renamed" = "1" ] && echo "y" || echo "ies" )"

echo ""
echo "ComAsset dataset downloaded to: $DATA_DIR"
echo "Categories:"
ls -1 "$DATA_DIR" | head -20
