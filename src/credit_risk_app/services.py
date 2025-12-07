# src/credit_risk_app/services.py
"""
Services module for Credit Risk Dashboard.

Contains:
- HFHubAssetManager: downloads assets from HuggingFace Hub at startup
- InferenceService: model inference with lazy caching
"""

import logging
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from sklearn import set_config
from fastapi import HTTPException

from .preprocessing import apply_transformations
from . import config

# Configure sklearn to preserve pandas DataFrames throughout pipelines
set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


# =============================================================================
# HF HUB ASSET MANAGER (NEW)
# =============================================================================


class HFHubAssetManager:
    """
    Manages downloading assets from HuggingFace Hub repositories.

    Downloads:
    - MLflow model from model repository
    - Raw data, SHAP files, and plots from data repository
    """

    @staticmethod
    def download_model() -> Path:
        """
        Download MLflow model from HF Hub.

        Returns:
            Path to the downloaded model directory
        """
        logger.info(f"[1/4] Downloading MLflow model from {config.HF_MODEL_REPO_ID}...")
        start = time.time()

        # Download entire model subdirectory
        snapshot_path = snapshot_download(
            repo_id=config.HF_MODEL_REPO_ID,
            allow_patterns=f"{config.HF_MODEL_SUBDIR}/**",
            cache_dir=config.HF_CACHE_DIR,
        )

        # The model is in a subdirectory
        model_source = Path(snapshot_path) / config.HF_MODEL_SUBDIR

        # Copy to expected local path
        config.LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        if config.LOCAL_MODEL_DIR.exists():
            shutil.rmtree(config.LOCAL_MODEL_DIR)
        shutil.copytree(model_source, config.LOCAL_MODEL_DIR)

        elapsed = time.time() - start
        logger.info(
            f"    Model downloaded in {elapsed:.2f}s -> {config.LOCAL_MODEL_DIR}"
        )

        return config.LOCAL_MODEL_DIR

    @staticmethod
    def download_data_file(
        filename: str, subfolder: Optional[str] = None, local_dir: Optional[Path] = None
    ) -> Path:
        """
        Download a single file from the data repository.

        Args:
            filename: Name of the file to download
            subfolder: Subfolder in the repo (e.g., "shap", "plots")
            local_dir: Local directory to save to

        Returns:
            Path to the downloaded file
        """
        repo_path = f"{subfolder}/{filename}" if subfolder else filename

        downloaded_path = hf_hub_download(
            repo_id=config.HF_DATA_REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            cache_dir=config.HF_CACHE_DIR,
        )

        # Copy to expected local path
        if local_dir:
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            shutil.copy2(downloaded_path, local_path)
            return local_path

        return Path(downloaded_path)

    @classmethod
    def download_all_assets(cls) -> dict:
        """
        Download all required assets from HF Hub.

        Returns:
            Dictionary with paths to all downloaded assets
        """
        logger.info("=" * 60)
        logger.info("DOWNLOADING ASSETS FROM HUGGING FACE HUB")
        logger.info("=" * 60)

        total_start = time.time()
        paths = {}

        # 1. Download model
        paths["model"] = cls.download_model()

        # 2. Download raw data
        logger.info("[2/4] Downloading raw data...")
        paths["data"] = cls.download_data_file(
            config.RAW_DATA_FILENAME, local_dir=config.LOCAL_DATA_DIR
        )
        logger.info(f"    -> {paths['data']}")

        # 3. Download SHAP files
        logger.info("[3/4] Downloading SHAP files...")
        paths["shap"] = cls.download_data_file(
            config.SHAP_EXPLANATION_FILENAME,
            subfolder="shap",
            local_dir=config.LOCAL_SHAP_DIR,
        )
        logger.info(f"    -> {paths['shap']}")

        # Also download SHAP beeswarm plot
        try:
            cls.download_data_file(
                config.SHAP_BEESWARM_FILENAME,
                subfolder="shap",
                local_dir=config.LOCAL_SHAP_DIR,
            )
            logger.info(
                f"    -> {config.LOCAL_SHAP_DIR / config.SHAP_BEESWARM_FILENAME}"
            )
        except Exception as e:
            logger.warning(f"    Could not download SHAP beeswarm: {e}")

        # 4. Download plot JSON files
        logger.info("[4/4] Downloading plot files...")
        paths["plots"] = []
        for plot_file in config.PLOT_FILENAMES:
            try:
                plot_path = cls.download_data_file(
                    plot_file, subfolder="plots", local_dir=config.LOCAL_PLOTS_DIR
                )
                paths["plots"].append(plot_path)
            except Exception as e:
                logger.warning(f"    Could not download {plot_file}: {e}")
        logger.info(f"    -> {len(paths['plots'])} plot files downloaded")

        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info(f"ALL ASSETS DOWNLOADED in {total_elapsed:.2f}s")
        logger.info("=" * 60)

        return paths


# =============================================================================
# INFERENCE SERVICE
# =============================================================================


class InferenceService:
    """
    Service for runtime inference with caching.

    Strategy:
    - Download assets from HF Hub at startup
    - Load model + raw data
    - Lazy preprocessing + prediction on first request (~5s warmup)
    - Cache all results in memory (instant subsequent requests)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        raw_data_path: Optional[Path] = None,
        shap_explanation_path: Optional[Path] = None,
        download_from_hub: bool = True,
    ):
        """
        Initialize the inference service.

        Args:
            model_path: Path to MLflow model (defaults to config.LOCAL_MODEL_DIR)
            raw_data_path: Path to raw data CSV (defaults to config.get_raw_data_path())
            shap_explanation_path: Path to SHAP joblib (defaults to config.get_shap_explanation_path())
            download_from_hub: If True, download assets from HF Hub first
        """
        # Download assets from HF Hub if enabled
        if download_from_hub:
            HFHubAssetManager.download_all_assets()

        # Set paths
        model_path = model_path or config.LOCAL_MODEL_DIR
        raw_data_path = raw_data_path or config.get_raw_data_path()
        shap_explanation_path = (
            shap_explanation_path or config.get_shap_explanation_path()
        )

        # 1. Load MLflow model
        logger.info(f"Loading MLflow model from {model_path}...")
        self.model = mlflow.pyfunc.load_model(str(model_path))
        self.expected_features = self.model.metadata.get_input_schema().input_names()
        self.threshold = float(
            self.model.metadata.metadata.get("optimal_threshold", 0.5)
        )
        logger.info(f"Model loaded. Threshold: {self.threshold}")

        # 2. Load raw data
        logger.info(f"Loading raw data from {raw_data_path}...")
        self.raw_data = pd.read_csv(
            raw_data_path, usecols=config.COLUMNS_TO_IMPORT, index_col="SK_ID_CURR"
        )
        logger.info(f"Loaded {len(self.raw_data)} clients")

        # 3. Load SHAP explanation (pre-computed)
        logger.info(f"Loading SHAP explanation from {shap_explanation_path}...")
        self.shap_explanation = joblib.load(shap_explanation_path)
        logger.info("SHAP explanation loaded")

        # 4. Initialize cache (lazy loading)
        self._predictions_cache: Optional[pd.DataFrame] = None
        self._dashboard_features = [
            "EXT_SOURCE_3",
            "EXT_SOURCE_2",
            "DAYS_EMPLOYED",
            "OWN_CAR_AGE",
        ]

    def _ensure_predictions_cached(self) -> None:
        """
        Lazy cache initialization: compute predictions on first request.
        """
        if self._predictions_cache is not None:
            return  # Already cached

        logger.info("WARMUP: computing predictions for all clients...")
        start = time.time()

        # Apply preprocessing
        X_processed = apply_transformations(
            self.raw_data.copy(), self.expected_features
        )

        # Generate predictions
        predictions = self.model.predict(X_processed)
        predictions_df = pd.DataFrame(
            predictions,
            index=self.raw_data.index,
            columns=["probability_neg", "probability_pos"],
        )

        # Prepare dashboard features (clean DAYS_EMPLOYED)
        dashboard_df = self.raw_data[self._dashboard_features].copy()
        dashboard_df["DAYS_EMPLOYED"] = (
            dashboard_df["DAYS_EMPLOYED"].replace(365243, np.nan).abs()
        )

        # Combine everything
        self._predictions_cache = dashboard_df.join(predictions_df)
        self._predictions_cache["threshold"] = self.threshold
        self._predictions_cache["decision"] = np.where(
            self._predictions_cache["probability_pos"] >= self.threshold,
            "refused",
            "accepted",
        )

        elapsed = time.time() - start
        logger.info(f"WARMUP COMPLETE in {elapsed:.2f}s. Cache ready.")

    def get_all_customer_ids(self) -> list[int]:
        """Returns list of all customer IDs (no cache needed)."""
        return self.raw_data.index.tolist()

    def _get_customer_data(self, customer_id: int) -> pd.Series:
        """
        Retrieve cached data for a customer.
        Triggers lazy cache initialization on first call.
        """
        self._ensure_predictions_cached()

        if customer_id not in self._predictions_cache.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )

        return self._predictions_cache.loc[customer_id]

    def get_score_data(self, customer_id: int) -> Dict[str, Any]:
        """Returns score data for gauge widget."""
        customer_data = self._get_customer_data(customer_id)
        return {
            "probability_pos": float(customer_data["probability_pos"]),
            "threshold": float(customer_data["threshold"]),
            "decision": customer_data["decision"],
        }

    def get_main_features(self, customer_id: int) -> Dict[str, Any]:
        """Returns the 4 main dashboard features."""
        customer_data = self._get_customer_data(customer_id)
        features = self._dashboard_features
        customer_features = customer_data[features].replace({np.nan: None})
        return customer_features.to_dict()

    def get_local_shap_values(self, customer_id: int) -> Dict[str, Any]:
        """Extracts local SHAP values (from pre-computed explanation)."""
        if customer_id not in self.raw_data.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )

        try:
            positional_idx = self.raw_data.index.get_loc(customer_id)
            shap_values = self.shap_explanation[positional_idx]
            return {
                "base_value": float(shap_values.base_values),
                "values": [float(v) for v in shap_values.values],
                "feature_names": shap_values.feature_names,
            }
        except Exception as e:
            logger.error(f"Error retrieving SHAP values for {customer_id}: {e}")
            raise HTTPException(
                status_code=500, detail="Could not retrieve SHAP values."
            )

    def get_bivariate_data(self, feat_x: str, feat_y: str) -> Dict[str, Any]:
        """Returns bivariate scatter data."""
        self._ensure_predictions_cached()

        if feat_x == feat_y:
            data_series = self._predictions_cache[feat_x].dropna()
            return {
                "x_data": data_series.tolist(),
                "y_data": data_series.tolist(),
            }

        bivariate_df = self._predictions_cache[[feat_x, feat_y]].dropna()
        return {
            "x_data": bivariate_df[feat_x].tolist(),
            "y_data": bivariate_df[feat_y].tolist(),
        }
