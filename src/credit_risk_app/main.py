# src/credit_risk_app/main.py
"""
Credit Risk Dashboard API, FastAPI backend with HF Hub asset download at startup.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse

from .services import InferenceService
from . import config

# --- Configuration & setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Lifespan & app initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Downloads assets from HF Hub and initializes the inference service at startup.
    """
    logger.info("Application startup: initializing resources...")
    config.print_config()

    try:
        # Initialize service (downloads assets from HF Hub automatically)
        service = InferenceService(download_from_hub=True)
        app.state.inference_service = service
        logger.info("Application startup complete. Service is ready.")
        logger.info(
            "Predictions will be computed on first dashboard request (warmup ~5-10s)"
        )
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}", exc_info=True)
        app.state.inference_service = None
    yield
    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan, title="Credit Risk API")


def get_inference_service(request: Request) -> InferenceService:
    """Dependency to get the inference service from app state."""
    if not request.app.state.inference_service:
        raise HTTPException(status_code=503, detail="Service is unavailable.")
    return request.app.state.inference_service


# =============================================================================
# API ENDPOINTS (ALL ORIGINAL ENDPOINTS PRESERVED)
# =============================================================================


@app.get("/customers", tags=["Dashboard Data"])
async def customers(service: InferenceService = Depends(get_inference_service)):
    """Returns a list of all available customer IDs."""
    return {"customer_ids": service.get_all_customer_ids()}


@app.get("/customer/{customer_id}/dashboard", tags=["Dashboard Composite"])
async def get_dashboard_data(
    customer_id: int, service: InferenceService = Depends(get_inference_service)
):
    """
    Composite endpoint returning all dashboard data in a single request.

    Note: First request triggers warmup (5-10s). Subsequent requests: <50ms.
    """
    return {
        "score": service.get_score_data(customer_id),
        "features": service.get_main_features(customer_id),
        "shap": service.get_local_shap_values(customer_id),
        "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
    }


@app.get("/customer/{customer_id}/score", tags=["Dashboard Widgets"])
async def get_score(
    customer_id: int, service: InferenceService = Depends(get_inference_service)
):
    """Endpoint for the score gauge widget."""
    return service.get_score_data(customer_id)


@app.get("/customer/{customer_id}/features", tags=["Dashboard Widgets"])
async def get_features(
    customer_id: int, service: InferenceService = Depends(get_inference_service)
):
    """Endpoint for the main features display."""
    return service.get_main_features(customer_id)


@app.get("/customer/{customer_id}/shap", tags=["Dashboard Widgets"])
async def get_shap_values(
    customer_id: int, service: InferenceService = Depends(get_inference_service)
):
    """Endpoint for the local SHAP importance (waterfall) plot."""
    return service.get_local_shap_values(customer_id)


@app.get("/features/bivariate_data", tags=["Dashboard Widgets"])
async def get_bivariate_data(
    feat_x: str, feat_y: str, service: InferenceService = Depends(get_inference_service)
):
    """Endpoint for the bi-variate analysis scatter plot."""
    return service.get_bivariate_data(feat_x, feat_y)


# =============================================================================
# STATIC ASSETS ENDPOINTS
# =============================================================================


@app.get("/shap/global", tags=["Static Assets"])
async def get_global_shap_plot():
    """
    Returns the pre-computed global SHAP beeswarm plot (PNG image).

    This image is generated once during model training and shows
    feature importance across all customers.
    """
    shap_beeswarm_path = config.get_shap_beeswarm_path()
    if not shap_beeswarm_path.exists():
        raise HTTPException(status_code=404, detail="SHAP beeswarm plot not found.")
    return FileResponse(
        shap_beeswarm_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/features/{feature_name}/distribution", tags=["Static Assets"])
async def get_feature_distribution(feature_name: str):
    """Returns pre-computed histogram data for a feature (JSON)."""
    if feature_name not in config.DISTRIBUTION_FEATURES:
        raise HTTPException(
            status_code=404,
            detail=f"Distribution not available for '{feature_name}'. "
            f"Available: {sorted(config.DISTRIBUTION_FEATURES)}",
        )
    file_path = config.LOCAL_PLOTS_DIR / f"{feature_name}_hist_data.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Distribution file not found.")
    with open(file_path, "r") as f:
        data = json.load(f)
    return JSONResponse(
        content=data,
        headers={"Cache-Control": "public, max-age=86400"},
    )
