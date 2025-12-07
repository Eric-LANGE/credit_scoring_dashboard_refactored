# src/credit_risk_app/preprocessing.py
import logging
import numpy as np
import pandas as pd
from typing import List

# Initialize a logger for this module
logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)


def apply_transformations(
    df: pd.DataFrame, expected_features: List[str]
) -> pd.DataFrame:
    """
    Orchestrates preprocessing steps on credit application data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing raw credit application data.
        This DataFrame is expected to contain columns loaded based on COLUMNS_TO_IMPORT.

    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame with columns matching expected_features.
        SK_ID_CURR as the index.
    """
    df_processed = df.copy()
    logger.debug(
        f"Input data shape for preprocessing: {df_processed.shape}, Index: {df_processed.index.name}"
    )

    _replace_placeholders(df_processed)
    _fill_missing_values(df_processed)
    _convert_time_columns(df_processed)
    _fix_region_rating(df_processed)
    _standardize_categoricals(df_processed)
    _engineer_ratio_features(df_processed)
    _cast_numeric_to_float(df_processed)

    logger.debug(
        f"Data shape after transformations, before final selection: {df_processed.shape}"
    )
    logger.debug(
        f"Available columns after transformations: {list(df_processed.columns)}"
    )

    try:
        df_final = df_processed[expected_features]
    except KeyError as e:
        missing_cols = [
            col for col in expected_features if col not in df_processed.columns
        ]
        available_cols = list(df_processed.columns)
        error_message = (
            f"Critical error in preprocessing: after all transformations, "
            f"the following expected features were not found: {missing_cols}. "
            f"Available columns after processing: {available_cols}. "
            f"Input df columns were: {list(df.columns)}. "
            f"Expected features list: {expected_features}."
        )
        logger.error(error_message)
        raise ValueError(error_message) from e

    logger.debug(
        f"Final data shape after selection: {df_final.shape}. Columns: {list(df_final.columns)}"
    )
    return df_final


# --- Helper functions ---


def _replace_placeholders(df: pd.DataFrame) -> None:
    """
    Replace known placeholder values with NaN.
    """
    placeholder_map = {
        "DAYS_EMPLOYED": 365243,
        "ORGANIZATION_TYPE": "XNA",
    }
    for col, placeholder in placeholder_map.items():
        if col in df.columns:
            if df[col].dtype == object or pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace(placeholder, np.nan)
                logger.debug(
                    f"Replaced placeholder '{placeholder}' in column '{col}' with NaN."
                )
            else:
                logger.warning(
                    f"Column '{col}' has unexpected type {df[col].dtype}, skipping placeholder replacement."
                )


def _fill_missing_values(df: pd.DataFrame) -> None:
    """
    Fill specific categorical columns with 'Unknown' where missing.
    """
    cols_to_fill = [
        col
        for col in (
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
        )
        if col in df.columns and df[col].dtype == object
    ]
    for col in cols_to_fill:
        original_nan_count = df[col].isna().sum()
        if original_nan_count > 0:
            df[col] = df[col].fillna("Unknown")
            logger.debug(
                f"Filled {original_nan_count} missing values in column '{col}' with 'Unknown'."
            )


def _convert_time_columns(df: pd.DataFrame) -> None:
    """
    Convert negative day counts to absolute values.
    """
    time_cols = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]
    present_time_cols = [c for c in time_cols if c in df.columns]
    if present_time_cols:
        for col in present_time_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].abs()
            else:
                logger.warning(
                    f"Column '{col}' expected to be numeric for time conversion but is {df[col].dtype}."
                )
    logger.debug(f"Applied absolute value to time columns: {present_time_cols}")


def _fix_region_rating(df: pd.DataFrame) -> None:
    if "REGION_RATING_CLIENT_W_CITY" in df.columns:
        if pd.api.types.is_numeric_dtype(df["REGION_RATING_CLIENT_W_CITY"]):
            original_count = (df["REGION_RATING_CLIENT_W_CITY"] == -1).sum()
            if original_count > 0:
                df["REGION_RATING_CLIENT_W_CITY"] = df[
                    "REGION_RATING_CLIENT_W_CITY"
                ].replace(-1, 2)
                logger.debug(
                    f"Fixed {original_count} instances of -1 to 2 in 'REGION_RATING_CLIENT_W_CITY'."
                )
        else:
            logger.warning(
                "Column 'REGION_RATING_CLIENT_W_CITY' expected to be numeric for -1 fix."
            )


def _standardize_categoricals(df: pd.DataFrame) -> None:
    """
    Use mapping dictionaries to make categorical values human-readable or consistent.

    """
    mappings = {
        "CODE_GENDER": {
            "M": "male",
            "F": "female",
            "XNA": "male",
        },
        "FLAG_OWN_CAR": {
            "Y": "yes",
            "N": "no",
        },
        "FLAG_OWN_REALTY": {"Y": "yes", "N": "no"},
        "FLAG_MOBIL": {1: "yes", 0: "no"},
        "FLAG_EMP_PHONE": {1: "yes", 0: "no"},
        "FLAG_WORK_PHONE": {1: "yes", 0: "no"},
        "FLAG_CONT_MOBILE": {1: "yes", 0: "no"},
        "FLAG_PHONE": {1: "yes", 0: "no"},
        "FLAG_EMAIL": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_3": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_6": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_8": {1: "yes", 0: "no"},
        # Region/City flags
        "REG_REGION_NOT_LIVE_REGION": {1: "different", 0: "same"},
        "REG_REGION_NOT_WORK_REGION": {1: "different", 0: "same"},
        "LIVE_REGION_NOT_WORK_REGION": {1: "different", 0: "same"},
        "REG_CITY_NOT_LIVE_CITY": {1: "different", 0: "same"},
        "REG_CITY_NOT_WORK_CITY": {1: "different", 0: "same"},
        "LIVE_CITY_NOT_WORK_CITY": {1: "different", 0: "same"},
        "REGION_RATING_CLIENT_W_CITY": {
            1: "A",
            2: "B",
            3: "C",
        },
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or (
                pd.api.types.is_integer_dtype(df[col])
                and all(isinstance(k, int) for k in mapping.keys())
            ):
                df[col] = df[col].replace(mapping)
            else:
                logger.debug(
                    f"Skipping standardization for column '{col}' due to dtype {df[col].dtype} or mapping key mismatch."
                )
    logger.debug("Attempted standardization mappings for categorical columns.")


def _engineer_ratio_features(df: pd.DataFrame) -> None:
    """
    Create new financial ratio features.
    """
    ratios = [
        ("PAYMENT_RATE", "AMT_ANNUITY", "AMT_CREDIT"),
        ("ANNUITY_INCOME_PERC", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("INCOME_CREDIT_PERC", "AMT_INCOME_TOTAL", "AMT_CREDIT"),
        ("DEBT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("CREDIT_PER_PERSON", "AMT_CREDIT", "CNT_FAM_MEMBERS"),
    ]
    for name, num_col, den_col in ratios:
        if num_col in df.columns and den_col in df.columns:
            # Ensure columns are numeric before division
            if pd.api.types.is_numeric_dtype(
                df[num_col]
            ) and pd.api.types.is_numeric_dtype(df[den_col]):
                denominator = df[den_col].replace(0, np.nan)
                df[name] = df[num_col] / denominator
                logger.debug(
                    f"Engineered feature '{name}' from '{num_col}' / '{den_col}'."
                )
            else:
                logger.warning(
                    f"Cannot engineer feature '{name}': Columns '{num_col}' or '{den_col}' are not numeric."
                )
        else:
            logger.debug(
                f"Skipping feature '{name}': Columns '{num_col}' or '{den_col}' not found."
            )


def _cast_numeric_to_float(df: pd.DataFrame) -> None:
    """
    Cast all numeric columns to float64 (MLflow signature requirement).
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        if df[col].dtype != "float64":
            df[col] = df[col].astype("float64")

    logger.debug(
        f"Cast {len([c for c in numeric_cols if df[c].dtype == 'float64'])} "
        f"numeric columns to float64"
    )
