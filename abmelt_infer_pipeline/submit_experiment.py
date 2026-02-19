#!/usr/bin/env python3

"""
CLI tool for submitting AbMelt inference experiments to Hugging Face Jobs.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import run_job, HfApi
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


def validate_sequences(heavy: str, light: str) -> bool:
    """Validate that sequences contain only valid amino acid codes."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    heavy_clean = heavy.upper().replace(" ", "")
    light_clean = light.upper().replace(" ", "")
    
    if not heavy_clean or not light_clean:
        raise ValueError("Sequences cannot be empty")
    
    invalid_heavy = set(heavy_clean) - valid_aa
    invalid_light = set(light_clean) - valid_aa
    
    if invalid_heavy:
        raise ValueError(f"Invalid amino acids in heavy chain: {invalid_heavy}")
    if invalid_light:
        raise ValueError(f"Invalid amino acids in light chain: {invalid_light}")
    
    return True


def validate_config(config_path: str) -> dict:
    """Validate and load config file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}")


def get_docker_image() -> str:
    """Get Docker image name from environment or use default."""
    return os.environ.get(
        "ABMELT_DOCKER_IMAGE",
        "your-dockerhub/abmelt:latest"  # User should set this
    )


def get_repo_url() -> str:
    """Get repository URL from environment or use default."""
    return os.environ.get(
        "REPO_URL",
        "https://github.com/Praful932/antibody-developability-abmelt"
    )


def get_hf_token() -> str:
    """Get HF token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is required. "
            "Set it with: export HF_TOKEN='hf_...'"
        )
    return token


def submit_experiment(
    name: str,
    config: str,
    heavy: Optional[str] = None,
    light: Optional[str] = None,
    pdb: Optional[str] = None,
    flavor: str = "a100-large",
    timeout: str = "24h",
    description: Optional[str] = None,
    docker_image: Optional[str] = None,
    repo_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    main_dataset: Optional[str] = None,
    detailed_dataset_prefix: Optional[str] = None,
    namespace: str = "hugging-science",
    skip_structure: bool = False,
    skip_md: bool = False,
    skip_descriptors: bool = False,
    skip_inference: bool = False,
    skip_detailed_dataset: bool = False,
    results_dir: Optional[str] = None,
    simulation_time: Optional[float] = None
):
    """
    Submit an experiment to Hugging Face Jobs.

    Args:
        name: Antibody name
        config: Path to config file
        heavy: Heavy chain sequence (required if pdb not provided)
        light: Light chain sequence (required if pdb not provided)
        pdb: Path to PDB file (alternative to heavy/light sequences)
        flavor: Hardware flavor (default: a100-large)
        timeout: Job timeout (e.g., "2h", "90m")
        description: Optional experiment description
        namespace: HF Jobs namespace (default: hugging-science)
        docker_image: Docker image name (defaults to env var or default)
        repo_url: Repository URL (defaults to env var or default)
        hf_token: HF token (defaults to HF_TOKEN env var)
        main_dataset: Main dataset name (defaults to env var)
        detailed_dataset_prefix: Detailed dataset prefix (defaults to env var)
        simulation_time: Override simulation_time from config (in nanoseconds)
    """
    # Validate inputs
    print("Validating inputs...")
    if pdb is not None:
        pdb_path = Path(pdb)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb}")
        if not pdb_path.is_file():
            raise ValueError(f"PDB path is not a file: {pdb}")
    elif heavy is not None and light is not None:
        validate_sequences(heavy, light)
    else:
        raise ValueError("Either --pdb or both --heavy and --light must be provided")
    validate_config(config)

    # Get defaults
    if docker_image is None:
        docker_image = get_docker_image()
    if repo_url is None:
        repo_url = get_repo_url()
    if hf_token is None:
        hf_token = get_hf_token()

    # Build command
    command = [
        "/hf_job_run.sh",
        "python", "run_experiment.py",
        "--name", name,
        "--config", config
    ]

    # Add sequence or PDB args
    if pdb is not None:
        staging_repo = f"{namespace}/abmelt-inputs"
        pdb_hub_filename = f"{name}.pdb"
        print(f"  Uploading PDB to HF Hub: {staging_repo}/{pdb_hub_filename}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=staging_repo, repo_type="dataset", exist_ok=True, private=False)
        if not api.file_exists(repo_id=staging_repo, filename=pdb_hub_filename, repo_type="dataset"):
            api.upload_file(
                path_or_fileobj=pdb,
                path_in_repo=pdb_hub_filename,
                repo_id=staging_repo,
                repo_type="dataset"
            )
            print(f"  Uploaded PDB to {staging_repo}/{pdb_hub_filename}")
        else:
            print(f"  PDB already exists in {staging_repo}/{pdb_hub_filename}, reusing.")
        command.extend(["--pdb-hub-repo", staging_repo, "--pdb-hub-file", pdb_hub_filename])
    else:
        command.extend(["--heavy", heavy, "--light", light])
    
    # Add skip flags if specified
    if skip_structure:
        command.append("--skip-structure")
    if skip_md:
        command.append("--skip-md")
    if skip_descriptors:
        command.append("--skip-descriptors")
    if skip_inference:
        command.append("--skip-inference")
    if results_dir:
        command.extend(["--results-dir", results_dir])
    
    # Add simulation_time override if specified
    if simulation_time is not None:
        command.extend(["--simulation-time", str(simulation_time)])
    
    # Prepare environment variables
    env_vars = {
        "REPO_URL": repo_url,
    }
    
    if description:
        env_vars["EXPERIMENT_DESCRIPTION"] = description
    
    if main_dataset:
        env_vars["HF_MAIN_DATASET"] = main_dataset
    
    if detailed_dataset_prefix:
        env_vars["HF_DETAILED_DATASET_PREFIX"] = detailed_dataset_prefix
    
    if skip_detailed_dataset:
        env_vars["SKIP_DETAILED_DATASET"] = "1"
    
    # Prepare secrets
    secrets = {
        "HF_TOKEN": hf_token
    }
    
    print(f"\nSubmitting experiment to HF Jobs...")
    print(f"  Antibody: {name}")
    print(f"  Hardware: {flavor}")
    print(f"  Timeout: {timeout}")
    print(f"  Config: {config}")
    print(f"  Docker Image: {docker_image}")
    print(f"  Namespace: {namespace}")
    print()
    
    # Submit job
    try:
        job = run_job(
            image=docker_image,
            command=command,
            flavor=flavor,
            timeout=timeout,
            namespace=namespace,
            env=env_vars,
            secrets=secrets
        )
        
        print("=" * 80)
        print("Job submitted successfully!")
        print(f"  Job ID: {job.id}")
        print(f"  Job URL: {job.url}")
        print("=" * 80)
        print("\nMonitor your job:")
        print(f"  python -c \"from huggingface_hub import inspect_job; print(inspect_job('{job.id}'))\"")
        print(f"\nView logs:")
        print(f"  python -c \"from huggingface_hub import fetch_job_logs; [print(log) for log in fetch_job_logs('{job.id}')]\"")
        print()
        
        return job
        
    except Exception as e:
        print(f"\nError submitting job: {e}")
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Submit AbMelt inference experiments to Hugging Face Jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic submission
  python submit_experiment.py \\
    --name "alemtuzumab" \\
    --heavy "QVQLQESGPGLVR..." \\
    --light "DIQMTQSPSSLSA..." \\
    --config configs/testing_config.yaml

  # With custom hardware and timeout
  python submit_experiment.py \\
    --name "test_ab" \\
    --heavy "QVQLQESGPGLVR..." \\
    --light "DIQMTQSPSSLSA..." \\
    --config configs/paper_config.yaml \\
    --flavor a10g-large \\
    --timeout 4h \\
    --description "Testing longer simulation time"
        """
    )
    
    parser.add_argument('--name', type=str, required=True,
                       help='Antibody name/identifier')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdb', type=str, default=None,
                             help='Path to PDB file (alternative to --heavy/--light)')
    input_group.add_argument('--heavy', '--h', type=str, default=None,
                             help='Heavy chain amino acid sequence (requires --light)')

    parser.add_argument('--light', '--l', type=str, default=None,
                       help='Light chain amino acid sequence (required when --heavy is used)')
    
    parser.add_argument('--flavor', type=str, default='a100-large',
                       help='Hardware flavor (default: a100-large)')
    parser.add_argument('--timeout', type=str, default='24h',
                       help='Job timeout (e.g., "24h", "90m", "3600s") (default: 24h)')
    parser.add_argument('--description', type=str, default=None,
                       help='Optional experiment description')
    
    parser.add_argument('--namespace', type=str, default='hugging-science',
                       help='HF Jobs namespace (default: hugging-science)')
    
    parser.add_argument('--docker-image', type=str, default=None,
                       help='Docker image name (defaults to ABMELT_DOCKER_IMAGE env var)')
    parser.add_argument('--repo-url', type=str, default=None,
                       help='Repository URL (defaults to REPO_URL env var)')
    
    parser.add_argument('--main-dataset', type=str, default=None,
                       help='Main dataset name (defaults to HF_MAIN_DATASET env var)')
    parser.add_argument('--detailed-dataset-prefix', type=str, default=None,
                       help='Detailed dataset prefix (defaults to HF_DETAILED_DATASET_PREFIX env var)')
    
    # Skip step flags
    parser.add_argument('--no-detailed-dataset', action='store_true',
                       help='Skip creating per-experiment detailed dataset')
    parser.add_argument('--skip-structure', action='store_true',
                       help='Skip structure preparation step')
    parser.add_argument('--skip-md', action='store_true',
                       help='Skip MD simulation step')
    parser.add_argument('--skip-descriptors', action='store_true',
                       help='Skip descriptor computation step')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip model inference step')
    
    # Debug/results directory
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to pre-computed results directory (for debugging with skip flags)')
    
    # Config overrides
    parser.add_argument('--simulation-time', type=float, default=None,
                       help='Override simulation_time from config (in nanoseconds, e.g., 0.5, 1, 2, 100). Minimum: ~0.1ns, Practical minimum: 1-2ns')
    
    args = parser.parse_args()

    # Validate: --heavy requires --light
    if args.heavy is not None and args.light is None:
        parser.error("--light is required when --heavy is provided")

    try:
        submit_experiment(
            name=args.name,
            heavy=args.heavy,
            light=args.light,
            pdb=args.pdb,
            config=args.config,
            flavor=args.flavor,
            timeout=args.timeout,
            description=args.description,
            namespace=args.namespace,
            docker_image=args.docker_image,
            repo_url=args.repo_url,
            main_dataset=args.main_dataset,
            detailed_dataset_prefix=args.detailed_dataset_prefix,
            skip_structure=args.skip_structure,
            skip_md=args.skip_md,
            skip_descriptors=args.skip_descriptors,
            skip_inference=args.skip_inference,
            skip_detailed_dataset=args.no_detailed_dataset,
            results_dir=args.results_dir,
            simulation_time=args.simulation_time
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
