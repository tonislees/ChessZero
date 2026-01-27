#!/usr/bin/env bash
set -uo pipefail

TARGET_DIR="./data/raw"
mkdir -p "$TARGET_DIR"
BASE_URL="https://www.pgnmentor.com/files.html"
HTML_TMP=$(mktemp)

echo "Fetching file list from $BASE_URL..."
curl -sSf "$BASE_URL" -o "$HTML_TMP"

echo "Parsing ZIP links..."
mapfile -t links < <(grep -oP 'href="\K[^"]*players/[^"]+\.zip' "$HTML_TMP" | sort -u || true)

if [ ${#links[@]} -eq 0 ]; then
    echo "No .zip links found — site format may have changed."
    rm "$HTML_TMP"
    exit 1
fi

echo "Found ${#links[@]} ZIP files. Downloading and extracting..."

total=${#links[@]}
count=0


for link in "${links[@]}"; do
    ((count++))

    if (( count == 2 )); then
      break
    fi

    # Resolve URL
    if [[ "$link" =~ ^http ]]; then
        url="$link"
    else
        url="https://www.pgnmentor.com/$link"
    fi

    fname=$(basename "$link")
    dl_path="$TARGET_DIR/$fname"

    # Progress display
    progress=$(( count * 100 / total ))
    bar_len=$(( progress / 2 ))
    bar=$(printf "%0.s#" $(seq 1 $bar_len))
    empty=$(printf "%0.s-" $(seq 1 $((50 - bar_len))))
    printf "\r[%s%s] %d%% (%d/%d)" "$bar" "$empty" "$progress" "$count" "$total"

    # Download and unzip
    curl -sSf "$url" -o "$dl_path"
    unzip -qqj "$dl_path" -d "$TARGET_DIR"
    rm "$dl_path"
done

echo "All files downloaded and extracted."
rm "$HTML_TMP"
