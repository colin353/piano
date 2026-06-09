#!/bin/sh
# Fetch the Steinway reference samples used by the scoring harness.
set -e
cd "$(dirname "$0")"
if [ -d assets/reference/SplendidGrandPiano ]; then
    echo "already fetched"
    exit 0
fi
git clone --depth 1 https://github.com/sfzinstruments/SplendidGrandPiano \
    assets/reference/SplendidGrandPiano
