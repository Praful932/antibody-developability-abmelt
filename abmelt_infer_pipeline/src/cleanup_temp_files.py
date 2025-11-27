#!/usr/bin/env python3

"""
Cleanup utility for AbMelt inference pipeline temp directory.
Removes intermediate GROMACS files while preserving files needed for:
- Re-running descriptor computation (--skip-md)
- Re-running inference (--skip-descriptors)
- Debugging and validation
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Dict
import glob

logger = logging.getLogger(__name__)


# Files that MUST be kept for pipeline to work
REQUIRED_FILES = {
    # Structure files
    "structure": [
        "{antibody_name}.pdb",  # Input structure
        "processed.pdb",  # Processed structure (needed for re-indexing)
        "processed.gro",  # Processed GRO (needed for re-indexing)
        "topol.top",  # Topology file
        "index.ndx",  # Index file for CDR regions
    ],
    
    # MD simulation final outputs (needed for descriptor computation)
    "md_final": [
        "md_final_{temp}.xtc",  # Final processed trajectory per temperature
        "md_final_{temp}.gro",  # Final reference structure per temperature
        "md_{temp}.tpr",  # Topology file per temperature
    ],
    
    # Descriptor computation outputs
    "descriptors": [
        "descriptors.csv",  # Aggregated descriptors (CSV)
        "descriptors.pkl",  # Aggregated descriptors (pickle)
        "*.xvg",  # All GROMACS descriptor files (needed for re-aggregation)
        "res_sasa_{temp}.np",  # SASA data per temperature
        "sconf_{temp}.log",  # Conformational entropy log per temperature
    ],
    
    # Order parameter files (optional - can be regenerated but useful for debugging)
    "order_params": [
        "order_s2_{temp}K_{block}_{start}.csv",  # Order parameter CSVs
        "order_lambda_{block}_{start}.csv",  # Lambda CSV
    ],
    
    # Model inference outputs
    "predictions": [
        "{antibody_name}_predictions.csv",  # Prediction results
    ],
}


# File patterns that can be safely deleted (intermediate files)
INTERMEDIATE_PATTERNS = [
    # GROMACS backup files (created when overwriting existing files)
    "#*#",  # Matches #filename.number# backup files
    
    # Intermediate trajectory processing files
    "md_whole_*.xtc",
    "md_nopbcjump_*.xtc",
    "md_{temp}.xtc",  # Raw trajectory before processing
    "md_{temp}.gro",  # Raw structure before processing
    
    # Equilibration files
    "nvt_*.gro",
    "nvt_*.xtc",
    "nvt_*.tpr",
    "nvt_*.cpt",
    "nvt_*.edr",
    "nvt_*.log",
    "npt_*.gro",
    "npt_*.xtc",
    "npt_*.tpr",
    "npt_*.cpt",
    "npt_*.edr",
    "npt_*.log",
    
    # System setup intermediates
    "box.gro",
    "solv.gro",
    "solv_ions.gro",
    "em.gro",
    "em.tpr",
    "em.edr",
    "em.log",
    "ions.tpr",
    
    # Covariance analysis intermediates
    "md_final_covar_*.xtc",
    "covar_*.trr",
    "covar_*.xvg",
    "covar.log",
    "avg_covar*.pdb",
    "covar_matrix_*.dat",
    
    # Custom simulation time files (when simulation_time != 100)
    "md_{temp}_*.gro",  # e.g., md_300_2.gro
    "md_{temp}_*.xtc",  # e.g., md_300_2.xtc
    "md_{temp}_*.tpr",  # e.g., md_300_2.tpr
    
    # Other intermediate files
    "em.trr",  # Energy minimization trajectory
    "hbond.ndx",  # Temporary index file for hydrogen bonds
    
    # PropKa output
    "*.pka",
    
    # MD run logs (except sconf logs which are kept)
    "md_*.log",
    "md_*.edr",
    
    # Checkpoint files (can be regenerated)
    "*.cpt",
    
    # Temporary MDP files created in work dir
    "nvt_*.mdp",
    "npt_*.mdp",
    "md_*.mdp",
    "mdout.mdp",  # GROMACS output MDP file
    "ions.mdp",
    "em.mdp",
    
    # Topology include files (generated during pdb2gmx)
    "posre_*.itp",  # Position restraint files
    "topol_*.itp",  # Topology include files for chains
]


def get_required_files(work_dir: Path, antibody_name: str, temperatures: List[str]) -> Set[str]:
    """
    Generate set of required file patterns based on antibody name and temperatures.
    
    Args:
        work_dir: Working directory path
        antibody_name: Name of antibody
        temperatures: List of temperature strings (e.g., ['300', '350', '400'])
        
    Returns:
        Set of required file paths (absolute)
    """
    required = set()
    
    # Structure files
    for pattern in REQUIRED_FILES["structure"]:
        file_path = work_dir / pattern.format(antibody_name=antibody_name)
        required.add(str(file_path))
    
    # MD final outputs (per temperature)
    for temp in temperatures:
        for pattern in REQUIRED_FILES["md_final"]:
            file_path = work_dir / pattern.format(temp=temp)
            required.add(str(file_path))
    
    # Descriptor outputs (per temperature)
    for temp in temperatures:
        for pattern in REQUIRED_FILES["descriptors"]:
            if "{temp}" in pattern:
                file_path = work_dir / pattern.format(temp=temp)
                required.add(str(file_path))
            elif pattern == "*.xvg":
                # Add all XVG files
                xvg_files = list(work_dir.glob("*.xvg"))
                required.update(str(f) for f in xvg_files)
            elif pattern == "res_sasa_{temp}.np":
                file_path = work_dir / pattern.format(temp=temp)
                required.add(str(file_path))
            elif pattern == "sconf_{temp}.log":
                file_path = work_dir / pattern.format(temp=temp)
                required.add(str(file_path))
    
    # Descriptor files (not temperature-specific)
    for pattern in ["descriptors.csv", "descriptors.pkl"]:
        file_path = work_dir / pattern
        required.add(str(file_path))
    
    # Order parameter files (optional - match any)
    order_s2_files = list(work_dir.glob("order_s2_*.csv"))
    order_lambda_files = list(work_dir.glob("order_lambda_*.csv"))
    required.update(str(f) for f in order_s2_files)
    required.update(str(f) for f in order_lambda_files)
    
    # Prediction files
    for pattern in REQUIRED_FILES["predictions"]:
        file_path = work_dir / pattern.format(antibody_name=antibody_name)
        required.add(str(file_path))
    
    return required


def get_intermediate_files(work_dir: Path, temperatures: List[str]) -> Set[str]:
    """
    Find all intermediate files that can be deleted.
    
    Args:
        work_dir: Working directory path
        temperatures: List of temperature strings
        
    Returns:
        Set of intermediate file paths (absolute)
    """
    intermediate = set()
    
    # Convert patterns to actual file matches
    for pattern in INTERMEDIATE_PATTERNS:
        # Handle temperature-specific patterns
        if "{temp}" in pattern:
            for temp in temperatures:
                actual_pattern = pattern.format(temp=temp)
                matches = list(work_dir.glob(actual_pattern))
                intermediate.update(str(f) for f in matches)
        else:
            # Special handling for GROMACS backup files (#*# pattern)
            if pattern == "#*#":
                # Match files that start with # and end with #
                # This is a glob pattern that matches GROMACS backup files
                matches = []
                for f in work_dir.rglob("*"):
                    if f.is_file() and f.name.startswith("#") and f.name.endswith("#"):
                        matches.append(f)
                intermediate.update(str(f) for f in matches)
            else:
                matches = list(work_dir.glob(pattern))
                intermediate.update(str(f) for f in matches)
    
    return intermediate


def cleanup_temp_directory(
    work_dir: Path,
    antibody_name: str,
    temperatures: List[str],
    dry_run: bool = True,
    keep_order_params: bool = True
) -> Dict[str, int]:
    """
    Clean up temporary directory, removing intermediate files.
    
    Args:
        work_dir: Working directory to clean
        antibody_name: Name of antibody
        temperatures: List of temperature strings
        dry_run: If True, only report what would be deleted without deleting
        keep_order_params: If True, keep order parameter CSV files
        
    Returns:
        Dictionary with cleanup statistics
    """
    work_dir = Path(work_dir).resolve()
    
    if not work_dir.exists():
        raise ValueError(f"Work directory does not exist: {work_dir}")
    
    # Get required files
    required = get_required_files(work_dir, antibody_name, temperatures)
    
    # Get intermediate files
    intermediate = get_intermediate_files(work_dir, temperatures)
    
    # Remove order param files from intermediate if keeping them
    if keep_order_params:
        order_param_files = set(str(f) for f in work_dir.glob("order_*.csv"))
        intermediate -= order_param_files
    
    # Find all files in directory
    all_files = set(str(f) for f in work_dir.rglob("*") if f.is_file())
    
    # Files to delete = intermediate files that are not required
    to_delete = intermediate - required
    
    # Also check for any other files not in required set (safety check)
    # But exclude hidden files and common non-GROMACS files
    other_files = all_files - required - intermediate
    suspicious = set()
    for f in other_files:
        f_path = Path(f)
        # Keep common non-GROMACS files
        if f_path.suffix in ['.py', '.yaml', '.yml', '.txt', '.md', '.json']:
            continue
        # Keep hidden files
        if f_path.name.startswith('.'):
            continue
        # Keep prediction CSV files (may have different naming conventions)
        if 'prediction' in f_path.name.lower() and f_path.suffix == '.csv':
            continue
        suspicious.add(f)
    
    stats = {
        "total_files": len(all_files),
        "required_files": len(required),
        "intermediate_files": len(intermediate),
        "files_to_delete": len(to_delete),
        "suspicious_files": len(suspicious),
    }
    
    if dry_run:
        logger.info("DRY RUN - No files will be deleted")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Required files: {stats['required_files']}")
        logger.info(f"Files to delete: {stats['files_to_delete']}")
        if suspicious:
            logger.warning(f"Suspicious files (not in required or intermediate): {len(suspicious)}")
            logger.warning("These files will NOT be deleted. Review manually:")
            for f in sorted(suspicious)[:10]:  # Show first 10
                logger.warning(f"  {Path(f).name}")
    else:
        # Actually delete files
        deleted_count = 0
        failed_count = 0
        
        for file_path in sorted(to_delete):
            try:
                Path(file_path).unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                failed_count += 1
        
        stats["deleted"] = deleted_count
        stats["failed"] = failed_count
        
        logger.info(f"Cleanup completed:")
        logger.info(f"  Deleted: {deleted_count} files")
        logger.info(f"  Failed: {failed_count} files")
        logger.info(f"  Remaining: {stats['total_files'] - deleted_count} files")
    
    return stats


def main():
    """CLI entry point for cleanup utility."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up AbMelt temp directory, removing intermediate GROMACS files"
    )
    parser.add_argument(
        "work_dir",
        type=str,
        help="Path to working directory (temp directory)"
    )
    parser.add_argument(
        "--antibody-name",
        type=str,
        required=True,
        help="Antibody name (for finding prediction files)"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        nargs="+",
        default=["300", "350", "400"],
        help="List of temperatures used in simulation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--delete-order-params",
        action="store_true",
        help="Also delete order parameter CSV files (default: keep them)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run cleanup
    stats = cleanup_temp_directory(
        work_dir=Path(args.work_dir),
        antibody_name=args.antibody_name,
        temperatures=args.temperatures,
        dry_run=args.dry_run,
        keep_order_params=not args.delete_order_params
    )
    
    print("\n" + "="*60)
    print("Cleanup Summary")
    print("="*60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main()

