import sys
import logging
import yaml
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from structure_prep import prepare_structure
from md_simulation import run_md_simulation
from compute_descriptors import compute_descriptors

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AbMelt Inference Pipeline')
    
    # Input options - either sequences or PDB file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--h', '--heavy', type=str, 
                           help='Heavy chain amino acid sequence (use with --l)')
    input_group.add_argument('--pdb', type=str,
                           help='Input PDB file path')
    
    parser.add_argument('--l', '--light', type=str,
                       help='Light chain amino acid sequence (use with --h)')
    parser.add_argument('--name', type=str, default='antibody',
                       help='Antibody name/identifier')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.h and not args.l:
        parser.error("--l/--light is required when using --h/--heavy")
    if args.l and not args.h:
        parser.error("--h/--heavy is required when using --l/--light")
    
    # 1. Load configuration
    config = load_config(args.config)
    
    # 2. Setup logging and directories
    setup_logging(config)
    create_directories(config)

    # 3. Create antibody input based on input type
    if args.pdb:
        # Input from PDB file
        antibody = {
            "name": args.name,
            "pdb_file": args.pdb,
            "type": "pdb"
        }
    else:
        # Input from sequences
        antibody = {
            "name": args.name,
            "heavy_chain": args.h,
            "light_chain": args.l,
            "type": "sequences"
        }
    
    # 4. Run structure preparation
    result = run_inference_pipeline(antibody, config)
    
    print(f"Inference pipeline for {args.name}:")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  PDB file: {result['structure_files']['pdb_file']}")
    print(f"  Work directory: {result['structure_files']['work_dir']}")
    
    if 'chains' in result['structure_files']:
        print(f"  Chains found: {list(result['structure_files']['chains'].keys())}")
    
    if 'simulation_result' in result:
        print(f"  MD simulations completed at temperatures: {list(result['simulation_result']['trajectory_files'].keys())}")
        for temp, files in result['simulation_result']['trajectory_files'].items():
            print(f"    {temp}K: {files['final_xtc']}")
    
    if 'descriptor_result' in result:
        print(f"  Descriptors computed: {result['descriptor_result']['descriptors_df'].shape[1]} features")
        print(f"  XVG files generated: {len(result['descriptor_result']['xvg_files'])}")
    
    return result
    


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise Exception(f"Failed to load config: {e}")

def setup_logging(config: dict):
    """Setup logging configuration."""
    log_level = getattr(logging, config["logging"]["level"].upper())
    log_file = config["logging"]["file"]
    
    # Create log directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directories(config: dict):
    """Create necessary directories."""
    script_directory = Path(__file__).parent.resolve()
    config['paths']['output_dir'] = script_directory / config["paths"]["run_dir"] / config['paths']['output_dir']
    config['paths']['temp_dir'] = script_directory / config["paths"]["run_dir"] / config['paths']['temp_dir']
    config['paths']['log_dir'] = script_directory / config["paths"]["run_dir"] / config['paths']['log_dir']

    directories = [
        config["paths"]["output_dir"],
        config["paths"]["temp_dir"],
        config["paths"]["log_dir"]
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_inference_pipeline(antibody, config):
    """Run the complete inference pipeline."""
    logging.info(f"Starting inference pipeline for antibody: {antibody['name']}")
    
    try:
        # Step 1: Structure preparation
        logging.info("Step 1: Preparing structure...")
        structure_files = prepare_structure(antibody, config)
        logging.info("Structure preparation completed")
        
        # Log structure files created
        logging.info(f"Structure files created:")
        for key, path in structure_files.items():
            if key != "chains":
                logging.info(f"  {key}: {path}")

        
        if "chains" in structure_files:
            logging.info(f"  chains: {list(structure_files['chains'].keys())}")
        
        # Step 2: MD simulation
        logging.info("Step 2: Running MD simulations...")
        simulation_result = run_md_simulation(structure_files, config)
        logging.info("MD simulations completed")
        
        # Log trajectory files created
        logging.info(f"Trajectory files created:")
        for temp, files in simulation_result["trajectory_files"].items():
            logging.info(f"  {temp}K: {files['final_xtc']}")
        
        # Step 3: Descriptor computation
        logging.info("Step 3: Computing descriptors...")
        descriptor_result = compute_descriptors(simulation_result, config)
        logging.info("Descriptor computation completed")
        
        # Log descriptor computation results
        logging.info(f"Descriptors computed:")
        logging.info(f"  DataFrame shape: {descriptor_result['descriptors_df'].shape}")
        logging.info(f"  Number of features: {len(descriptor_result['descriptors_df'].columns)}")
        logging.info(f"  XVG files generated: {len(descriptor_result['xvg_files'])}")
        
        # TODO: Step 4: ML prediction
        
        result = {
            "status": "success",
            "structure_files": structure_files,
            "simulation_result": simulation_result,
            "descriptor_result": descriptor_result,
            "message": "Descriptor computation completed. Ready for ML prediction."
        }
        
        logging.info("Inference pipeline completed successfully")
        return result
        
    except Exception as e:
        logging.error(f"Inference pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()