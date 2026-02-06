#!/usr/bin/env python3

"""
Experiment runner wrapper for AbMelt inference pipeline.
Handles experiment tracking, result collection, and HF dataset uploads.
"""

import sys
import os
import argparse
import logging
import yaml
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from infer import run_inference_pipeline, load_config, setup_logging, create_directories
from dataset_utils import (
    upload_to_main_predictions_dataset,
    upload_to_detailed_results_dataset,
    get_hf_token
)

logger = logging.getLogger(__name__)


def generate_experiment_id() -> str:
    """Generate unique experiment ID based on timestamp."""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_job_id() -> str:
    """Get HF Jobs ID from environment variable."""
    return os.environ.get("HF_JOB_ID", "unknown")


def get_experiment_description() -> Optional[str]:
    """Get experiment description from environment variable."""
    return os.environ.get("EXPERIMENT_DESCRIPTION", None)


def get_git_info() -> Dict[str, str]:
    """Get Git commit information for metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        return {
            "git_commit": commit_hash,
            "git_branch": branch
        }
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
        return {"git_commit": "unknown", "git_branch": "unknown"}


def run_experiment(
    antibody_name: str,
    heavy_chain: str,
    light_chain: str,
    config_path: str,
    hf_token: Optional[str] = None,
    skip_structure: bool = False,
    skip_md: bool = False,
    skip_descriptors: bool = False,
    skip_inference: bool = False,
    results_dir: Optional[str] = None
) -> Dict:
    """
    Run a complete inference experiment and upload results to HF datasets.
    
    Args:
        antibody_name: Name of the antibody
        heavy_chain: Heavy chain amino acid sequence
        light_chain: Light chain amino acid sequence
        config_path: Path to configuration YAML file
        hf_token: HF token (optional, will use HF_TOKEN env var if not provided)
        
    Returns:
        Dictionary with experiment results and status
    """
    start_time = datetime.now()
    experiment_id = generate_experiment_id()
    job_id = get_job_id()
    
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Antibody: {antibody_name}")
    logger.info(f"Job ID: {job_id}")
    logger.info("=" * 80)
    
    # Get HF token
    if hf_token is None:
        hf_token = get_hf_token()
    
    if not hf_token:
        raise ValueError("HF_TOKEN is required for dataset uploads")
    
    # Load config
    try:
        config = load_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
    
    # Setup logging and directories
    setup_logging(config)
    create_directories(config)
    
    # Override temp_dir with results_dir if specified (for debugging with skip flags)
    # Do this AFTER create_directories() so absolute paths aren't modified
    if results_dir:
        logger.info(f"Using pre-computed results from: {results_dir}")
        config["paths"]["temp_dir"] = results_dir
    
    # Prepare antibody input
    antibody = {
        "name": antibody_name,
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "type": "sequences"
    }
    
    # Run inference pipeline
    status = "success"
    error_message = None
    result = None
    predictions = None
    descriptors_df = None
    log_file = None
    
    try:
        logger.info("Running inference pipeline...")
        result = run_inference_pipeline(
            antibody, 
            config,
            skip_structure=skip_structure,
            skip_md=skip_md,
            skip_descriptors=skip_descriptors,
            skip_inference=skip_inference
        )
        
        # Extract results
        if "inference_result" in result:
            predictions = result["inference_result"]["predictions"]
            logger.info("Successfully obtained predictions")
        else:
            logger.warning("No inference result found")
            predictions = {"tagg": None, "tm": None, "tmon": None}
        
        if "descriptor_result" in result:
            descriptors_df = result["descriptor_result"]["descriptors_df"]
            logger.info(f"Successfully obtained descriptors: {descriptors_df.shape}")
        else:
            logger.warning("No descriptor result found")
            descriptors_df = None
        
        # Get log file path
        log_file = config.get("logging", {}).get("file", None)
        if log_file:
            # Resolve relative to run_dir
            run_dir = Path(config["paths"]["run_dir"])
            log_file = str(run_dir / log_file)
            # Check if log file exists
            if not Path(log_file).exists():
                logger.warning(f"Log file not found: {log_file}")
                log_file = None
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        logger.error(traceback.format_exc())
        status = "failed"
        error_message = str(e)
        
        # Try to extract partial results if available
        if result:
            if "inference_result" in result:
                predictions = result["inference_result"].get("predictions", {"tagg": None, "tm": None, "tmon": None})
            if "descriptor_result" in result:
                descriptors_df = result["descriptor_result"].get("descriptors_df", None)
    
    # Calculate duration
    end_time = datetime.now()
    duration_seconds = int((end_time - start_time).total_seconds())
    
    logger.info(f"Experiment duration: {duration_seconds} seconds")
    
    # Upload to main predictions dataset
    try:
        logger.info("Uploading to main predictions dataset...")
        upload_to_main_predictions_dataset(
            experiment_id=experiment_id,
            antibody_name=antibody_name,
            heavy_chain=heavy_chain,
            light_chain=light_chain,
            predictions=predictions or {"tagg": None, "tm": None, "tmon": None},
            config=config,
            job_id=job_id,
            status=status,
            duration_seconds=duration_seconds,
            error_message=error_message,
            token=hf_token
        )
        logger.info("Successfully uploaded to main predictions dataset")
    except Exception as e:
        logger.error(f"Failed to upload to main dataset: {e}")
        logger.error(traceback.format_exc())
        # Don't raise - we want to try uploading detailed results even if main fails
    
    # Upload to detailed results dataset (only if we have descriptors)
    if descriptors_df is not None:
        try:
            logger.info("Uploading to detailed results dataset...")
            
            # Prepare metadata
            metadata = {
                "experiment_id": experiment_id,
                "antibody_name": antibody_name,
                "job_id": job_id,
                "status": status,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration_seconds,
                "description": get_experiment_description(),
                **get_git_info()
            }
            
            if error_message:
                metadata["error_message"] = error_message
            
            upload_to_detailed_results_dataset(
                experiment_id=experiment_id,
                descriptors_df=descriptors_df,
                config=config,
                log_file=log_file,
                metadata=metadata,
                token=hf_token
            )
            logger.info("Successfully uploaded to detailed results dataset")
        except Exception as e:
            logger.error(f"Failed to upload to detailed dataset: {e}")
            logger.error(traceback.format_exc())
            # Don't raise - main upload succeeded
    
    # Print summary
    logger.info("=" * 80)
    logger.info(f"Experiment {experiment_id} completed")
    logger.info(f"Status: {status}")
    if status == "success":
        if predictions:
            logger.info("Predictions:")
            for model_name, pred in predictions.items():
                if pred is not None:
                    logger.info(f"  {model_name.upper()}: {pred[0]:.3f}")
                else:
                    logger.info(f"  {model_name.upper()}: FAILED")
    else:
        logger.info(f"Error: {error_message}")
    logger.info("=" * 80)
    
    return {
        "experiment_id": experiment_id,
        "status": status,
        "predictions": predictions,
        "duration_seconds": duration_seconds,
        "error_message": error_message
    }


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description='AbMelt Experiment Runner - Runs inference and uploads to HF datasets'
    )
    
    parser.add_argument('--name', type=str, required=True,
                       help='Antibody name/identifier')
    parser.add_argument('--heavy', '--h', type=str, required=True,
                       help='Heavy chain amino acid sequence')
    parser.add_argument('--light', '--l', type=str, required=True,
                       help='Light chain amino acid sequence')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file path')
    
    parser.add_argument('--skip-structure', action='store_true',
                       help='Skip structure preparation step')
    parser.add_argument('--skip-md', action='store_true',
                       help='Skip MD simulation step')
    parser.add_argument('--skip-descriptors', action='store_true',
                       help='Skip descriptor computation step')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip model inference step')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to pre-computed results directory (for debugging with skip flags)')
    
    args = parser.parse_args()
    
    # Run experiment
    try:
        result = run_experiment(
            antibody_name=args.name,
            heavy_chain=args.heavy,
            light_chain=args.light,
            config_path=args.config,
            skip_structure=args.skip_structure,
            skip_md=args.skip_md,
            skip_descriptors=args.skip_descriptors,
            skip_inference=args.skip_inference,
            results_dir=args.results_dir
        )
        
        # Exit with appropriate code
        if result["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
