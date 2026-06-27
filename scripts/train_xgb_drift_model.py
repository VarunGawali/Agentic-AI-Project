"""
Train XGBoost drift model for XGB-GARCH-t forecaster.

The model predicts next-period log return:

    y_t = log(P[t+1] / P[t])

using features from engines.features.

Outputs:
    models/xgb_drift_model.json
    models/xgb_feature_columns.json
    models/xgb_training_metadata.json

Run:
    uv run python -m scripts.train_xgb_drift_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Run: uv pip install xgboost"
    ) from exc

from hedging_assistant.data.loader import load_price_history
from hedging_assistant.engines.features import FEATURE_COLUMNS, make_supervised_dataset
from hedging_assistant.engines.xgb_garch_forecaster import upload_artifact_to_blob

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_xgb")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "xgb_drift_model.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "xgb_feature_columns.json"
METADATA_PATH = MODEL_DIR / "xgb_training_metadata.json"


def train_xgb_model(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    train_fraction: float = 0.85,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost drift model using a time-based train/validation split.
    """

    if X.empty:
        raise ValueError("Feature matrix X is empty.")

    if len(X) != len(y):
        raise ValueError(f"len(X)={len(X)} does not match len(y)={len(y)}")

    if not 0.5 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 and 1.0")

    split_idx = int(len(X) * train_fraction)

    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Invalid train/validation split.")

    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]

    X_val = X.iloc[split_idx:]
    y_val = y[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    metrics = {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "train_rmse": rmse(y_train, train_pred),
        "validation_rmse": rmse(y_val, val_pred),
        "train_mae": mae(y_train, train_pred),
        "validation_mae": mae(y_val, val_pred),
        "validation_directional_accuracy": directional_accuracy(y_val, val_pred),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y, ddof=1)),
        "prediction_mean_validation": float(np.mean(val_pred)),
        "prediction_std_validation": float(np.std(val_pred, ddof=1)),
    }

    return model, metrics


def save_artifacts(
    model: xgb.XGBRegressor,
    feature_columns: list[str],
    metadata: dict,
    upload: bool = False,
) -> None:
    """
    Save trained model, feature columns, and metadata locally.
    When upload=True (and AZURE_STORAGE_CONNECTION_STRING is set),
    all three artifacts are also pushed to Azure Blob Storage.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved artifacts locally")
    logger.info("  Model           : %s", MODEL_PATH)
    logger.info("  Feature columns : %s", FEATURE_COLUMNS_PATH)
    logger.info("  Metadata        : %s", METADATA_PATH)

    if upload:
        if not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            logger.warning(
                "--upload requested but AZURE_STORAGE_CONNECTION_STRING is not set; skipping."
            )
            return

        for local_path, blob_name_env, default_blob in [
            (MODEL_PATH, "XGB_MODEL_BLOB_NAME", "xgb_drift_model.json"),
            (FEATURE_COLUMNS_PATH, "XGB_FEATURE_COLUMNS_BLOB_NAME", "xgb_feature_columns.json"),
            (METADATA_PATH, "XGB_METADATA_BLOB_NAME", "xgb_training_metadata.json"),
        ]:
            blob_name = os.environ.get(blob_name_env, default_blob)
            upload_artifact_to_blob(local_path, blob_name)

        logger.info("All artifacts uploaded to Azure Blob Storage.")


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Measures how often model predicts the correct return direction.
    """

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost drift model for crude price forecasting."
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="WTI",
        choices=["WTI", "BRENT"],
        help="Crude symbol to train on.",
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="History start date.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.85,
        help="Time-based training fraction.",
    )

    parser.add_argument(
        "--inventory-csv",
        type=str,
        default=None,
        help="Optional CSV with date,eia_inventory_chg.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Rows to drop at beginning due to rolling features.",
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload trained artifacts to Azure Blob Storage after saving.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading price history...")
    df = load_price_history(
        symbol=args.symbol,
        start=args.start,
    )

    logger.info("Rows loaded : %d", len(df))
    logger.info("Date range  : %s to %s", df["date"].min(), df["date"].max())

    logger.info("Building supervised dataset...")
    X, y, target_df = make_supervised_dataset(
        df=df,
        inventory_csv=args.inventory_csv,
        warmup=args.warmup,
    )

    logger.info("Feature matrix shape : %s", X.shape)
    logger.info("Target shape         : %s", y.shape)
    logger.info("Feature columns      : %s", list(X.columns))

    missing_cols = set(FEATURE_COLUMNS) - set(X.columns)

    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    X = X[FEATURE_COLUMNS]

    logger.info("Training XGBoost drift model...")
    model, metrics = train_xgb_model(
        X=X,
        y=y,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )

    metadata = {
        "model_type": "xgb_drift",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "symbol": args.symbol,
        "start": args.start,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "warmup": args.warmup,
        "feature_columns": FEATURE_COLUMNS,
        "target": "next_period_log_return",
        "rows_used_after_warmup": int(len(X)),
        "first_target_date": str(target_df["date"].min().date()),
        "last_target_date": str(target_df["date"].max().date()),
        "metrics": metrics,
        "notes": [
            "XGBoost predicts conditional drift / expected next-period log return.",
            "GARCH-t forecaster will model volatility and fat-tailed residual shocks separately.",
            "eia_inventory_chg is optional and should be zero during forward simulation unless scenario values are supplied.",
        ],
    }

    save_artifacts(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        metadata=metadata,
        upload=args.upload,
    )

    logger.info("Metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.6f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    logger.info("Done.")


if __name__ == "__main__":
    main()