#!/usr/bin/env python3

"""
Model inference module for AbMelt pipeline.
Loads trained models and makes predictions on computed descriptors.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AbMeltPredictor:
    """
    Predictor class that loads trained models and makes predictions.
    """

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize the predictor with model directory.

        Args:
            models_dir: Path to directory containing trained models
        """
        if models_dir is None:
            # Default to local models directory
            current_dir = Path(__file__).parent
            models_dir = current_dir.parent / "models"

        self.models_dir = Path(models_dir)

        # Model paths
        self.model_paths = {
            "tagg": self.models_dir / "tagg" / "efs_best_knn.pkl",
            "tm": self.models_dir / "tm" / "efs_best_randomforest.pkl",
            "tmon": self.models_dir / "tmon" / "efs_best_elasticnet.pkl",
        }

        # Feature definitions for each model
        # These are the features selected by exhaustive feature selection
        self.model_features = {
            "tagg": [
                "rmsf_cdrs_mu_400",
                "gyr_cdrs_Rg_std_400",
                "all-temp_lamda_b=25_eq=20",
            ],
            "tm": [
                "gyr_cdrs_Rg_std_350",
                "bonds_contacts_std_350",
                "rmsf_cdrl1_std_350",
            ],
            "tmon": [
                "bonds_contacts_std_350",
                "all-temp-sasa_core_mean_k=20_eq=20",
                "all-temp-sasa_core_std_k=20_eq=20",
                "r-lamda_b=2.5_eq=20",
            ],
        }

        # Loaded models cache
        self.loaded_models: dict[str, Any] = {}

        # Validate model files exist
        self._validate_models()

    def _validate_models(self) -> None:
        """Validate that all model files exist."""
        missing_models = []
        for model_name, model_path in self.model_paths.items():
            if not model_path.exists():
                missing_models.append(f"{model_name}: {model_path}")

        if missing_models:
            raise FileNotFoundError(
                "Missing model files:\n" + "\n".join(missing_models)
            )

        logger.info(f"Validated {len(self.model_paths)} model files")

    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_name: Name of the model (tagg, tm, or tmon)

        Returns:
            Loaded scikit-learn model
        """
        if model_name not in self.model_paths:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(self.model_paths.keys())}"
            )

        # Check cache first
        if model_name in self.loaded_models:
            logger.debug(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]

        # Load model
        model_path = self.model_paths[model_name]
        logger.info(f"Loading model: {model_name} from {model_path}")

        try:
            model = joblib.load(model_path)
            self.loaded_models[model_name] = model
            logger.info(f"Successfully loaded {model_name} model")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def extract_features(
        self, descriptors_df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """
        Extract required features for a specific model.

        Args:
            descriptors_df: DataFrame containing all computed descriptors
            model_name: Name of the model

        Returns:
            DataFrame with only the features required by the model
        """
        required_features = self.model_features[model_name]

        # Check which features are available
        available_features = descriptors_df.columns.tolist()
        missing_features = [f for f in required_features if f not in available_features]

        if missing_features:
            logger.error(f"Missing features for {model_name}: {missing_features}")
            logger.error(f"Available features: {available_features}")
            raise ValueError(
                f"Missing required features for {model_name}: {missing_features}"
            )

        # Extract features in the correct order
        features_df = descriptors_df[required_features].copy()

        logger.info(f"Extracted {len(required_features)} features for {model_name}")
        logger.debug(f"Features: {required_features}")

        return features_df

    def predict(self, descriptors_df: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Make prediction using a specific model.

        Args:
            descriptors_df: DataFrame containing all computed descriptors
            model_name: Name of the model (tagg, tm, or tmon)

        Returns:
            Array of predictions
        """
        # Load model
        model = self.load_model(model_name)

        # Extract required features
        features_df = self.extract_features(descriptors_df, model_name)

        # Make prediction
        logger.info(f"Making prediction with {model_name} model...")
        predictions = model.predict(features_df)

        logger.info(f"Prediction completed: {predictions}")

        return np.asarray(predictions)

    def predict_all(self, descriptors_df: pd.DataFrame) -> dict[str, np.ndarray | None]:
        """
        Make predictions using all three models.

        Args:
            descriptors_df: DataFrame containing all computed descriptors

        Returns:
            Dictionary mapping model name to predictions
        """
        logger.info("Making predictions with all models...")

        predictions: dict[str, np.ndarray | None] = {}
        for model_name in ["tagg", "tm", "tmon"]:
            try:
                pred = self.predict(descriptors_df, model_name)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {e}")
                predictions[model_name] = None

        return predictions


def run_model_inference(descriptor_result: dict, config: dict) -> dict:
    """
    Main entry point for model inference.

    Args:
        descriptor_result: Dictionary containing descriptors DataFrame
        config: Configuration dictionary

    Returns:
        Dictionary containing predictions and metadata
    """
    logger.info("Starting model inference...")

    descriptors_df = descriptor_result["descriptors_df"]
    work_dir = Path(descriptor_result["work_dir"])
    antibody_name = work_dir.name

    # Initialize predictor
    try:
        predictor = AbMeltPredictor()
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise

    # Make predictions
    try:
        predictions = predictor.predict_all(descriptors_df)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

    # Format results
    results = _format_predictions(predictions, antibody_name)

    # Save predictions
    try:
        _save_predictions(results, work_dir, antibody_name)
    except Exception as e:
        logger.warning(f"Failed to save predictions: {e}")

    logger.info("Model inference completed successfully")

    return {
        "status": "success",
        "predictions": predictions,
        "results": results,
        "work_dir": str(work_dir),
        "message": "Model inference completed successfully",
    }


def _format_predictions(
    predictions: dict[str, np.ndarray | None], antibody_name: str
) -> pd.DataFrame:
    """
    Format predictions into a readable DataFrame.

    Args:
        predictions: Dictionary mapping model name to predictions
        antibody_name: Name of the antibody

    Returns:
        DataFrame with formatted results
    """
    results_data = {
        "antibody": [antibody_name],
    }

    # Add predictions for each target
    target_names = {"tagg": "T_agg (°C)", "tm": "T_m (°C)", "tmon": "T_m_onset (°C)"}

    for model_name, pred in predictions.items():
        if pred is not None:
            # Convert from standardized/scaled value to actual temperature
            # Note: These are still relative values unless you have the original scaling
            target_name = target_names.get(model_name, model_name)
            results_data[target_name] = [
                str(float(pred[0])) if len(pred) > 0 else "N/A"
            ]
        else:
            target_name = target_names.get(model_name, model_name)
            results_data[target_name] = ["N/A"]

    results_df = pd.DataFrame(results_data)

    logger.info("Prediction results:")
    logger.info(f"\n{results_df.to_string(index=False)}")

    return results_df


def _save_predictions(
    results: pd.DataFrame, work_dir: Path, antibody_name: str
) -> None:
    """
    Save predictions to files.

    Args:
        results: DataFrame containing formatted results
        work_dir: Working directory
        antibody_name: Name of the antibody
    """
    # Save to CSV
    csv_path = work_dir / f"{antibody_name}_predictions.csv"
    results.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions to {csv_path}")

    # Also save to a summary file in the results directory
    try:
        results_dir = work_dir.parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_path = results_dir / "predictions_summary.csv"

        # Append to summary file if it exists, otherwise create new
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            combined = pd.concat([existing, results], ignore_index=True)
            combined.to_csv(summary_path, index=False)
            logger.info(f"Appended predictions to {summary_path}")
        else:
            results.to_csv(summary_path, index=False)
            logger.info(f"Created predictions summary at {summary_path}")
    except Exception as e:
        logger.warning(f"Could not save to summary file: {e}")


def load_existing_predictions(work_dir: Path, antibody_name: str) -> dict:
    """
    Load previously computed predictions from disk.

    Args:
        work_dir: Working directory
        antibody_name: Name of the antibody

    Returns:
        Dictionary containing predictions
    """
    csv_path = work_dir / f"{antibody_name}_predictions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    logger.info(f"Loading existing predictions from {csv_path}")
    results_df = pd.read_csv(csv_path)

    # Extract predictions back into dictionary format
    predictions = {
        "tagg": np.array([results_df["T_agg (°C)"].values[0]])
        if "T_agg (°C)" in results_df.columns
        else None,
        "tm": np.array([results_df["T_m (°C)"].values[0]])
        if "T_m (°C)" in results_df.columns
        else None,
        "tmon": np.array([results_df["T_m_onset (°C)"].values[0]])
        if "T_m_onset (°C)" in results_df.columns
        else None,
    }

    return {
        "status": "success",
        "predictions": predictions,
        "results": results_df,
        "work_dir": str(work_dir),
        "message": "Loaded existing predictions",
    }
