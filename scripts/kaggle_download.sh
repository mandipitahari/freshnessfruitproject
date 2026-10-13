#!/usr/bin/env bash
set -e
mkdir -p data
kaggle datasets download -d raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables -p data
unzip -o data/fresh-and-stale-images-of-fruits-and-vegetables.zip -d data
