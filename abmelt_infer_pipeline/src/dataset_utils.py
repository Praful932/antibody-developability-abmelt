#!/usr/bin/env python3

"""
Dataset utilities for uploading AbMelt experiment results to Hugging Face datasets.
Handles both main predictions dataset (one row per experiment) and detailed results datasets.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

try:
    from datasets import Dataset, Features, load_dataset, Value
    from huggingface_hub import HfApi, upload_file, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")
    logging.error("Please install: pip install datasets huggingface_hub")
    raise

logger = logging.getLogger(__name__)

# Canonical descriptor columns from features_used.md (9 unique features across all models)
DESCRIPTOR_COLUMNS = [
    "gyr_cdrs_Rg_std_350",           # Tm
    "bonds_contacts_std_350",       # Tm, Tmon
    "rmsf_cdrl1_std_350",           # Tm
    "rmsf_cdrs_mu_400",             # Tagg
    "gyr_cdrs_Rg_std_400",          # Tagg
    "all-temp_lamda_b=25_eq=20",    # Tagg
    "all-temp-sasa_core_mean_k=20_eq=20",  # Tmon
    "all-temp-sasa_core_std_k=20_eq=20",   # Tmon
    "r-lamda_b=2.5_eq=20",          # Tmon
]


def _convert_paths_to_str(obj):
    """
    Recursively convert Path objects to strings for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, Path, or any JSON-serializable type)
        
    Returns:
        Object with all Path instances converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_str(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_str(v) for v in obj)
    return obj


def get_hf_token() -> Optional[str]:
    """Get HF token from environment variable."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not found in environment. Dataset uploads will fail.")
    return token


def login_to_hf(token: Optional[str] = None):
    """Login to Hugging Face Hub."""
    if token is None:
        token = get_hf_token()
    if token:
        try:
            login(token=token)
            logger.info("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face Hub: {e}")
            raise
    else:
        raise ValueError("HF_TOKEN is required for dataset uploads")


def compute_config_hash(config: Dict) -> str:
    """Compute MD5 hash of config dictionary for quick comparison."""
    # Convert Path objects to strings for JSON serialization
    serializable_config = _convert_paths_to_str(config)
    config_str = json.dumps(serializable_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def extract_trackable_params(config: Dict) -> Dict[str, Any]:
    """
    Extract all trackable parameters from config and flatten them.
    
    Returns a flat dictionary with all parameters that should be tracked in the dataset.
    """
    params = {}
    
    # Simulation parameters
    sim = config.get("simulation", {})
    params["temperatures"] = json.dumps(_convert_paths_to_str(sim.get("temperatures", [])))
    params["simulation_time"] = sim.get("simulation_time", None)
    params["force_field"] = sim.get("force_field", None)
    params["water_model"] = sim.get("water_model", None)
    params["salt_concentration"] = sim.get("salt_concentration", None)
    params["pH"] = sim.get("pH", None)
    params["p_salt"] = sim.get("p_salt", None)
    params["n_salt"] = sim.get("n_salt", None)
    params["gpu_enabled"] = sim.get("gpu_enabled", False)
    
    # GROMACS parameters
    gromacs = config.get("gromacs", {})
    params["gpu_id"] = gromacs.get("gpu_id", None)
    params["n_threads"] = gromacs.get("n_threads", None)
    
    # Descriptor parameters
    descriptors = config.get("descriptors", {})
    params["equilibration_time"] = descriptors.get("equilibration_time", None)
    params["block_length"] = json.dumps(_convert_paths_to_str(descriptors.get("block_length", [])))
    params["core_surface_k"] = descriptors.get("core_surface_k", None)
    params["compute_lambda"] = descriptors.get("compute_lambda", False)
    params["use_dummy_s2"] = descriptors.get("use_dummy_s2", False)
    
    # Performance parameters
    performance = config.get("performance", {})
    params["cleanup_temp"] = performance.get("cleanup_temp", False)
    params["cleanup_after"] = performance.get("cleanup_after", None)
    params["delete_order_params"] = performance.get("delete_order_params", False)
    params["save_trajectories"] = performance.get("save_trajectories", False)
    
    return params


def create_main_dataset_if_not_exists(
    dataset_name: str,
    token: Optional[str] = None
) -> bool:
    """
    Create the main predictions dataset if it doesn't exist.
    
    Best practice: First create the repo, then push data to it.
    This follows the recommended HuggingFace workflow:
    1. api.create_repo() to create the repository
    2. dataset.push_to_hub() to upload data
    
    Args:
        dataset_name: Full dataset name (e.g., "username/abmelt-experiments")
        token: HF token (optional, will use HF_TOKEN env var if not provided)
        
    Returns:
        True if dataset was created, False if it already exists
    """
    if token is None:
        token = get_hf_token()
    
    if not token:
        raise ValueError("HF_TOKEN is required")
    
    login_to_hf(token)
    
    # Initialize HfApi for repo operations
    api = HfApi(token=token)
    
    try:
        # Try to load the dataset
        _ = load_dataset(dataset_name, split="train")
        logger.info(f"Dataset {dataset_name} already exists")
        return False
    except Exception:
        # Dataset doesn't exist, create it with empty data
        logger.info(f"Creating new dataset: {dataset_name}")
        
        # STEP 1: Create the repository first (REQUIRED!)
        # This is the key fix - create_repo before push_to_hub
        try:
            logger.info(f"Step 1: Creating repository {dataset_name}...")
            api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                private=False,
                exist_ok=True  # Don't fail if somehow already exists
            )
            logger.info(f"Repository created successfully")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise
        
        # STEP 2: Define schema with empty data
        empty_data = {
            "experiment_id": [],
            "antibody_name": [],
            "timestamp": [],
            "heavy_chain": [],
            "light_chain": [],
            "tagg": [],
            "tm": [],
            "tmon": [],
            "job_id": [],
            "status": [],
            "duration_seconds": [],
            "config_hash": [],
            "git_commit": [],
            "error_message": [],
            "description": [],
            # Config parameters
            "temperatures": [],
            "simulation_time": [],
            "force_field": [],
            "water_model": [],
            "salt_concentration": [],
            "pH": [],
            "p_salt": [],
            "n_salt": [],
            "equilibration_time": [],
            "block_length": [],
            "core_surface_k": [],
            "compute_lambda": [],
            "use_dummy_s2": [],
            "cleanup_temp": [],
            "cleanup_after": [],
            "delete_order_params": [],
            "save_trajectories": [],
            "gpu_enabled": [],
            "gpu_id": [],
            "n_threads": [],
        }
        # Descriptor columns from features_used.md
        for col in DESCRIPTOR_COLUMNS:
            empty_data[col] = []

        # Define Features schema with float32 for numeric and descriptor columns
        features = Features({
            "experiment_id": Value("string"),
            "antibody_name": Value("string"),
            "timestamp": Value("string"),
            "heavy_chain": Value("string"),
            "light_chain": Value("string"),
            "tagg": Value("float32"),
            "tm": Value("float32"),
            "tmon": Value("float32"),
            "job_id": Value("string"),
            "status": Value("string"),
            "duration_seconds": Value("float32"),
            "config_hash": Value("string"),
            "git_commit": Value("string"),
            "error_message": Value("string"),
            "description": Value("string"),
            "temperatures": Value("string"),
            "simulation_time": Value("float32"),
            "force_field": Value("string"),
            "water_model": Value("string"),
            "salt_concentration": Value("float32"),
            "pH": Value("float32"),
            "p_salt": Value("string"),
            "n_salt": Value("string"),
            "equilibration_time": Value("float32"),
            "block_length": Value("string"),
            "core_surface_k": Value("float32"),
            "compute_lambda": Value("bool"),
            "use_dummy_s2": Value("bool"),
            "cleanup_temp": Value("bool"),
            "cleanup_after": Value("string"),
            "delete_order_params": Value("bool"),
            "save_trajectories": Value("bool"),
            "gpu_enabled": Value("bool"),
            "gpu_id": Value("float32"),
            "n_threads": Value("float32"),
            **{col: Value("float32") for col in DESCRIPTOR_COLUMNS},
        })

        # STEP 3: Push data to the repository
        logger.info(f"Step 2: Pushing initial data to {dataset_name}...")
        dataset = Dataset.from_dict(empty_data, features=features)
        dataset.push_to_hub(dataset_name, token=token)
        logger.info(f"Successfully created dataset: {dataset_name}")
        return True


def upload_to_main_predictions_dataset(
    experiment_id: str,
    antibody_name: str,
    heavy_chain: str,
    light_chain: str,
    predictions: Dict[str, Optional[np.ndarray]],
    config: Dict,
    job_id: str,
    status: str,
    duration_seconds: int,
    error_message: Optional[str] = None,
    dataset_name: Optional[str] = None,
    token: Optional[str] = None,
    descriptors_df: Optional[pd.DataFrame] = None,
    description: Optional[str] = None,
    git_commit: Optional[str] = None
):
    """
    Upload a single experiment result to the main predictions dataset.
    
    Args:
        experiment_id: Unique experiment identifier
        antibody_name: Name of the antibody
        heavy_chain: Heavy chain sequence
        light_chain: Light chain sequence
        predictions: Dictionary with keys "tagg", "tm", "tmon" (values are numpy arrays or None)
        config: Configuration dictionary used for this experiment
        job_id: HF Jobs ID
        status: "success", "failed", or "timeout"
        duration_seconds: Total runtime in seconds
        error_message: Optional error message if status is "failed"
        dataset_name: Full dataset name (defaults to HF_MAIN_DATASET env var or "username/abmelt-experiments")
        token: HF token (optional, will use HF_TOKEN env var if not provided)
        descriptors_df: DataFrame with computed descriptors (optional; missing features will be None)
        description: Optional experiment description (from --description / EXPERIMENT_DESCRIPTION)
        git_commit: Git commit hash of the code used for this experiment
    """
    if token is None:
        token = get_hf_token()
    
    if not token:
        raise ValueError("HF_TOKEN is required")
    
    if dataset_name is None:
        dataset_name = os.environ.get("HF_MAIN_DATASET", "Praful932/abmelt-experiments")
    logger.info(f"Main dataset name - {dataset_name}")
    
    login_to_hf(token)
    
    # Initialize HfApi for repo operations
    api = HfApi(token=token)
    
    # Ensure repository exists (best practice: always call this before push)
    try:
        logger.info(f"Ensuring repository exists: {dataset_name}")
        api.create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            private=False,
            exist_ok=True  # Don't fail if already exists
        )
        logger.info(f"Repository confirmed/created")
    except Exception as e:
        logger.warning(f"Could not ensure repo exists: {e}")
        # Continue anyway - it might already exist
    
    # Create dataset if it doesn't exist (this will also create if needed)
    create_main_dataset_if_not_exists(dataset_name, token)
    
    # Extract trackable parameters
    config_params = extract_trackable_params(config)
    config_hash = compute_config_hash(config)
    
    # Extract predictions
    tagg = float(predictions.get("tagg", [None])[0]) if predictions.get("tagg") is not None else None
    tm = float(predictions.get("tm", [None])[0]) if predictions.get("tm") is not None else None
    tmon = float(predictions.get("tmon", [None])[0]) if predictions.get("tmon") is not None else None
    
    # Prepare row data
    row_data = {
        "experiment_id": experiment_id,
        "antibody_name": antibody_name,
        "timestamp": datetime.now().isoformat(),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "tagg": tagg,
        "tm": tm,
        "tmon": tmon,
        "job_id": job_id,
        "status": status,
        "duration_seconds": duration_seconds,
        "config_hash": config_hash,
        "git_commit": git_commit or "",
        "error_message": error_message or "",
        "description": description or "",
        **config_params
    }
    # Add descriptor columns (use value from descriptors_df if present, else None)
    for col in DESCRIPTOR_COLUMNS:
        if descriptors_df is not None and col in descriptors_df.columns and len(descriptors_df) > 0:
            val = descriptors_df[col].iloc[0]
            row_data[col] = None if pd.isna(val) else float(val)
        else:
            row_data[col] = None
    
    # Load existing dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Loaded existing dataset with {len(dataset)} rows")
        df = dataset.to_pandas()
    except ValueError as e:
        # Handle empty dataset (0 rows) - "Instruction 'train' corresponds to no data!"
        if "corresponds to no data" in str(e):
            logger.warning(f"Dataset exists but is empty (0 rows), starting with empty DataFrame")
            # Create empty DataFrame with the proper schema
            df = pd.DataFrame(columns=list(row_data.keys()))
        else:
            logger.error(f"Failed to load dataset: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Append new row
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)

    # Cast descriptor columns to float32 for schema consistency
    for col in DESCRIPTOR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    
    # Convert back to Dataset and push
    new_dataset = Dataset.from_pandas(df)
    new_dataset.push_to_hub(dataset_name, token=token)
    
    logger.info(f"Successfully uploaded experiment {experiment_id} to {dataset_name}")


def upload_to_detailed_results_dataset(
    experiment_id: str,
    descriptors_df: pd.DataFrame,
    config: Dict,
    log_file: Optional[str] = None,
    metadata: Optional[Dict] = None,
    dataset_prefix: Optional[str] = None,
    token: Optional[str] = None
):
    """
    Upload detailed results (descriptors, config, logs) to a per-experiment dataset.
    
    Args:
        experiment_id: Unique experiment identifier
        descriptors_df: DataFrame with all computed descriptors
        config: Configuration dictionary used for this experiment
        log_file: Path to log file (optional)
        metadata: Additional metadata dictionary (optional)
        dataset_prefix: Dataset prefix (defaults to HF_DETAILED_DATASET_PREFIX env var)
        token: HF token (optional, will use HF_TOKEN env var if not provided)
    """
    if token is None:
        token = get_hf_token()
    
    if not token:
        raise ValueError("HF_TOKEN is required")
    
    if dataset_prefix is None:
        dataset_prefix = os.environ.get(
            "HF_DETAILED_DATASET_PREFIX",
            "Praful932/abmelt-experiments-"
        )
    
    dataset_name = f"{dataset_prefix}{experiment_id}"
    login_to_hf(token)
    
    api = HfApi(token=token)
    
    # Create a temporary directory for files to upload
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save descriptors as CSV
        descriptors_csv = temp_path / "descriptors.csv"
        descriptors_df.to_csv(descriptors_csv, index=False)
        logger.info(f"Saved descriptors CSV: {descriptors_csv}")
        
        # Save descriptors as pickle
        descriptors_pkl = temp_path / "descriptors.pkl"
        descriptors_df.to_pickle(descriptors_pkl)
        logger.info(f"Saved descriptors pickle: {descriptors_pkl}")
        
        # Save config as YAML
        import yaml
        config_yaml = temp_path / "config.yaml"
        with open(config_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config YAML: {config_yaml}")
        
        # Copy log file if provided
        log_dest = None
        if log_file and Path(log_file).exists():
            log_dest = temp_path / "inference.log"
            shutil.copy2(log_file, log_dest)
            logger.info(f"Copied log file: {log_file} -> {log_dest}")
        
        # Save metadata JSON
        if metadata is None:
            metadata = {}
        metadata_json = temp_path / "run_metadata.json"
        with open(metadata_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata JSON: {metadata_json}")
        
        # Upload all files
        files_to_upload = [
            (descriptors_csv, "descriptors.csv"),
            (descriptors_pkl, "descriptors.pkl"),
            (config_yaml, "config.yaml"),
            (metadata_json, "run_metadata.json"),
        ]
        
        if log_dest:
            files_to_upload.append((log_dest, "inference.log"))
        
        # Create repo if it doesn't exist (REQUIRED before uploading files!)
        try:
            logger.info(f"Creating repository {dataset_name}...")
            api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                exist_ok=True,  # Don't fail if already exists
                token=token,
                private=False
            )
            logger.info(f"Repository created/confirmed")
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")
            # Continue anyway - might already exist
        
        # Upload files
        for file_path, filename in files_to_upload:
            try:
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=token
                )
                logger.info(f"Uploaded {filename} to {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to upload {filename}: {e}")
                raise
    
    logger.info(f"Successfully uploaded detailed results to {dataset_name}")
