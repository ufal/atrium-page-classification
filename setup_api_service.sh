#!/bin/bash

# Configuration
VENV_DIR="venv"
REQUIREMENTS="requirements.txt"
SERV_REQUIREMENTS="service/requirements.txt"
MODEL_DIR="model"
# List of model versions to check/download
# Maps specific revision tags (e.g., v5.3) to expected folder names (e.g., model_v53)
declare -A MODELS=( 
    ["v1.3"]="model_v13" 
    ["v2.3"]="model_v23" 
    ["v3.3"]="model_v33" 
    ["v4.3"]="model_v43" 
    ["v5.3"]="model_v53" 
)

echo "🚀 Starting Atrium Page Classification Service Setup..."

# 1. Environment Setup
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment ($VENV_DIR)..."
    python3 -m venv $VENV_DIR
else
    echo "✅ Virtual environment found."
fi

# Activate environment
source $VENV_DIR/bin/activate

# 2. Install Dependencies
if [ -f "$REQUIREMENTS" ]; then
    echo "⬇️ Installing dependencies from $REQUIREMENTS..."
    pip install -r $REQUIREMENTS
else
    echo "⚠️ Warning: $REQUIREMENTS not found. Skipping pip install."
fi

# 2. Install Dependencies
if [ -f "$SERV_REQUIREMENTS" ]; then
    echo "⬇️ Installing server dependencies from $SERV_REQUIREMENTS..."
    pip install -r $SERV_REQUIREMENTS
else
    echo "⚠️ Warning: $SERV_REQUIREMENTS not found. Skipping pip install."
fi

# 3. Model Weights Download
echo "🧠 Checking model weights..."

# Ensure model directory exists
mkdir -p $MODEL_DIR

for VER in "${!MODELS[@]}"; do
    FOLDER_NAME="${MODELS[$VER]}"
    TARGET_PATH="$MODEL_DIR/$FOLDER_NAME"

    if [ ! -d "$TARGET_PATH" ]; then
        echo "⬇️ Model $VER not found at $TARGET_PATH. Downloading from HuggingFace via run.py..."
        
        # Check if run.py exists before running
        if [ -f "run.py" ]; then
            python3 run.py --hf -rev $VER
            
            if [ $? -eq 0 ]; then
                echo "✅ Successfully downloaded $VER."
            else
                echo "❌ Failed to download $VER."
            fi
        else
            echo "❌ Error: run.py not found in root. Cannot download model $VER."
        fi
    else
        echo "✅ Model $VER already exists ($FOLDER_NAME). Skipping."
    fi
done

echo "🎉 Setup complete! To start the server, run:"
echo "   source venv/bin/activate"
echo "   uvicorn service.api:app --reload"
