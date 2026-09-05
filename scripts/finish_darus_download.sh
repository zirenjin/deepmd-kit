#!/usr/bin/env bash
set -euo pipefail

dest=${1:?usage: finish_darus_download.sh DEST_DIR}
url=${2:?usage: finish_darus_download.sh DEST_DIR URL}
total=140595324
parts=8
chunk=$(((total + parts - 1) / parts))

download_part() {
    local i=$1
    local base=$((i * chunk))
    local end=$((base + chunk - 1))
    local expected
    local have
    local start
    [ "$end" -ge "$total" ] && end=$((total - 1))
    expected=$((end - base + 1))
    while :; do
        have=$(stat -c %s "$dest/part.$i" 2>/dev/null || printf '0')
        [ "$have" -ge "$expected" ] && [ "$have" -eq "$expected" ] && return
        start=$((base + have))
        curl --http1.1 -L --fail --retry 10 --range "$start-$end" \
            -o "$dest/part.$i.resume" "$url" \
            >>"$dest/part.$i.resume.log" 2>&1
        cat "$dest/part.$i.resume" >> "$dest/part.$i"
        rm -f "$dest/part.$i.resume"
    done
}

mkdir -p "$dest"
for i in $(seq 0 $((parts - 1))); do
    download_part "$i" &
done
wait

for i in $(seq 0 $((parts - 1))); do
    base=$((i * chunk))
    end=$((base + chunk - 1))
    [ "$end" -ge "$total" ] && end=$((total - 1))
    test "$(stat -c %s "$dest/part.$i")" -eq "$((end - base + 1))"
done

cat "$dest"/part.{0..7} > "$dest/direct_upsampling_data_rungs_1-3.zip"
sha256sum "$dest/direct_upsampling_data_rungs_1-3.zip"
unzip -t "$dest/direct_upsampling_data_rungs_1-3.zip" >/dev/null
printf '%s\n' ZIP_OK
