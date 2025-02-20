#!/bin/bash
set -e

MODEL_DIR="/root/.ollama/models"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' does not exist."
    exit 1
fi

echo "Setting up Ollama models in $MODEL_DIR..."

if ! ls "$MODEL_DIR"/*.gguf >/dev/null 2>&1; then
    echo "Error: No .gguf files found in $MODEL_DIR"
    exit 1
fi

for model in "$MODEL_DIR"/*.gguf; do
    if [ -f "$model" ]; then
        model_name=$(basename "$model" .gguf)
        model_folder="$MODEL_DIR/$model_name"

        mkdir -p "$model_folder"

        modelfile_path="$model_folder/Modelfile"
        if [ ! -f "$modelfile_path" ]; then
            echo "FROM /root/.ollama/models/$(basename "$model")" > "$modelfile_path"
            ollama create "$model_name" -f "$modelfile_path"
            echo "Setup Modelfile for $model_name"
        else
            echo "Modelfile already exists for $model_name, skipping..."
        fi
    fi
done

echo "Model setup complete"
