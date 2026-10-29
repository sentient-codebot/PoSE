#!/usr/bin/env bash
# Download the Fluvius 2024 15-minute dataset (Open data license - Fluvius;
# attribution: Fluvius, dataset last updated 2025-09-29).
# Dataset page: https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/
# Usage: bash scripts/fluvius/download_fluvius.sh [data/fluvius/raw]
set -euo pipefail
OUT="${1:-data/fluvius/raw}"
BASE="https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/attachments"
mkdir -p "$OUT"
fetch() {
    local id="$1" name="$2"
    if [ ! -s "$OUT/$name" ]; then
        curl -fL --retry 3 -o "$OUT/$name" "$BASE/$id"
    fi
}
fetch p6269_1_50_dmk_sample_elek_2024_deel_1_zip "P6269_1_50_DMK_Sample_Elek_2024_Deel_1.zip"
fetch p6269_1_50_dmk_sample_elek_2024_deel_2_zip "P6269_1_50_DMK_Sample_Elek_2024_Deel_2.zip"
fetch 1_50_aanleveirng_dmk_elek_fluvius_legende_25200929_xlsx \
    "1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx"
# hash the downloaded files only, not the checksum file itself
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$OUT" && sha256sum -- *.zip *.xlsx | tee SHA256SUMS.txt)
else
    (cd "$OUT" && shasum -a 256 -- *.zip *.xlsx | tee SHA256SUMS.txt)
fi
# Expected sizes (bytes, 2026-10-08): Deel_1 197375122, Deel_2 199377167,
# legend 20436 (SHA-256 344ac3a69adab39bb0e6cdf490b63380ea5ebb0c83cc26ab964c73430c1b0acf).
