# =============================================================================
# Credit Risk Dashboard
# =============================================================================
# Assets (model, data, SHAP, plots) are downloaded from HuggingFace Hub at runtime.
# This keeps the image small (~50MB vs ~150MB with embedded assets).
# =============================================================================

FROM mambaorg/micromamba:latest

WORKDIR /app

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace Hub settings
    HF_HOME=/tmp/.huggingface \
    HF_HUB_DISABLE_SYMLINKS=1

# Install dependencies (including huggingface_hub)
COPY --chown=$MAMBA_USER:$MAMBA_USER credit_risk_env.yml /app/
RUN micromamba install -y -n base -f /app/credit_risk_env.yml && \
    micromamba clean -afy --quiet

# Copy application code only (no data, models, shap, plots)
COPY --chown=$MAMBA_USER:$MAMBA_USER ./src /app/src

# Copy tests
COPY --chown=$MAMBA_USER:$MAMBA_USER ./tests /app/tests

# Create directories for runtime assets (populated by HF Hub download)
RUN mkdir -p /app/models /app/data /app/shap /app/plots

# Copy entrypoint
COPY --chown=$MAMBA_USER:$MAMBA_USER --chmod=755 entrypoint.sh /app/

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
