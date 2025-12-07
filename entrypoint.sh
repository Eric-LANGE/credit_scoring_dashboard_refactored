#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Entrypoint script for Credit Risk Dashboard
# Launches FastAPI (port 8000) + Streamlit (port 7860)
# =============================================================================

# Environment setup
export MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"
export MPLCONFIGDIR="/tmp/matplotlib"
export PYTHONPATH="/app/src:${PYTHONPATH:-}"

# Configure HF cache directory based on available storage
if [ -d "/data" ] && [ -w "/data" ]; then
    export HF_HOME="/data/.huggingface"
    mkdir -p "$HF_HOME"
    echo "Using persistent storage for HF cache: $HF_HOME"
else
    export HF_HOME="/tmp/.huggingface"
    mkdir -p "$HF_HOME"
    echo "WARNING: No persistent storage available, using temp dir: $HF_HOME"
fi

echo "===== Application startup at $(date +'%Y-%m-%d %H:%M:%S') ====="
echo "HF_HOME: $HF_HOME"
echo "HF_MODEL_REPO_ID: ${HF_MODEL_REPO_ID:-not set}"
echo "HF_DATA_REPO_ID: ${HF_DATA_REPO_ID:-not set}"

# Cleanup function
cleanup() {
    echo "--- Shutting down services ---"
    if [ ! -z "${FASTAPI_PID:-}" ]; then
        kill -TERM "$FASTAPI_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

if [ "${1:-}" = "pytest" ]; then
    echo "--- Running tests ---"
    exec pytest
else
    echo "--- Starting internal FastAPI service on port 8000 ---"
    uvicorn src.credit_risk_app.main:app --host 0.0.0.0 --port 8000 &
    FASTAPI_PID=$!

    # Give the API time to download assets and start up
    echo "Waiting for FastAPI to initialize (downloading assets from HF Hub)..."
    sleep 30

    echo "--- Starting public Streamlit dashboard on port 7860 ---"
    exec streamlit run src/credit_risk_app/dashboard.py \
        --server.port 7860 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false \
        --server.headless true \
        --server.fileWatcherType none
fi
