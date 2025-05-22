#!/bin/bash
set -e

echo "--- Installing dependencies ---"
pip install -r requirements.txt

echo "--- Checking model file ---"
if [ -f "model/model.pkl" ]; then
    echo "Model file exists"
else
    echo "ERROR: Model file not found!"
    exit 1
fi
