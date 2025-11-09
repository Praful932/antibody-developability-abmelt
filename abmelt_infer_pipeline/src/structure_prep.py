#!/usr/bin/env python3

"""
Structure preparation module for AbMelt inference pipeline.
Handles antibody structure generation and preprocessing.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add the original AbMelt src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "AbMelt" / "src"))

try:
    from structure import immune_builder
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

logger = logging.getLogger(__name__)


def prepare_structure(antibody: Dict, config: Dict) -> Dict[str, str]:
    """
    Prepare antibody structure for MD simulation.
    
    Args:
        antibody: Dictionary containing antibody information
        config: Configuration dictionary
        
    Returns:
        Dictionary with paths to prepared structure files
    """
    logger.info(f"Preparing structure for antibody: {antibody['name']}")
    
    if antibody["type"] == "pdb":
        return _prepare_from_pdb(antibody, config)
    elif antibody["type"] == "sequences":
        return _prepare_from_sequences(antibody, config)
    else:
        raise ValueError(f"Unsupported antibody type: {antibody['type']}")


def _prepare_from_pdb(antibody: Dict, config: Dict) -> Dict[str, str]:
    """Prepare structure from existing PDB file."""
    pdb_file = antibody["pdb_file"]
    antibody_name = antibody["name"]
    
    # Create working directory
    work_dir = Path(config["paths"]["temp_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy PDB to working directory
    import shutil
    target_pdb = work_dir / f"{antibody_name}.pdb"
    shutil.copy2(pdb_file, target_pdb)
    
    # Preprocess the structure
    return _preprocess_structure(str(target_pdb), antibody_name, work_dir, config)


def _prepare_from_sequences(antibody: Dict, config: Dict) -> Dict[str, str]:
    """Generate structure from heavy and light chain sequences."""
    heavy_chain = antibody["heavy_chain"]
    light_chain = antibody["light_chain"]
    antibody_name = antibody["name"]
    
    # Create working directory
    work_dir = Path(config["paths"]["temp_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate structure using ImmuneBuilder
    logger.info("Generating structure using ImmuneBuilder...")
    sequence = {'H': heavy_chain, 'L': light_chain}
    pdb_file = work_dir / f"{antibody_name}.pdb"
    
    try:
        immune_builder(sequence, str(pdb_file))
        logger.info(f"Structure generated: {pdb_file}")
    except Exception as e:
        logger.error(f"Failed to generate structure: {e}")
        raise
    
    # Preprocess the structure
    return _preprocess_structure(str(pdb_file), antibody_name, work_dir, config)


def _preprocess_structure(pdb_file: str, antibody_name: str, work_dir: Path, config: Dict) -> Dict[str, str]:
    """Basic structure validation and organization."""
    logger.info("Validating structure...")
    
    # Validate the PDB structure
    if not validate_structure(pdb_file):
        logger.error("Structure validation failed")
        raise Exception("Structure validation failed")
    
    # Extract chain sequences for reference
    chains = get_chain_sequences(pdb_file)
    logger.info(f"Found chains: {list(chains.keys())}")
    
    # Return basic structure information
    structure_files = {
        "pdb_file": str(work_dir / f"{antibody_name}.pdb"),
        "work_dir": str(work_dir),
        "chains": chains
    }
    
    logger.info("Structure preparation completed successfully")
    return structure_files


def validate_structure(pdb_file: str) -> bool:
    """Validate that the PDB file contains a proper antibody structure."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("antibody", pdb_file)
        
        # Check for heavy and light chains
        chains = list(structure.get_chains())
        if len(chains) < 2:
            logger.warning("Structure should contain at least 2 chains (heavy and light)")
            return False
        
        # Basic validation passed
        return True
        
    except Exception as e:
        logger.error(f"Structure validation failed: {e}")
        return False


def get_chain_sequences(pdb_file: str) -> Dict[str, str]:
    """Extract heavy and light chain sequences from PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("antibody", pdb_file)
        chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) 
                 for chain in structure.get_chains()}
        return chains
    except Exception as e:
        logger.error(f"Failed to extract sequences: {e}")
        return {}


# Convenience functions for direct usage
def generate_structure_from_sequences(heavy_chain: str, light_chain: str, 
                                    output_file: str = "antibody.pdb") -> str:
    """Generate antibody structure from sequences using ImmuneBuilder."""
    sequence = {'H': heavy_chain, 'L': light_chain}
    immune_builder(sequence, output_file)
    return output_file


def prepare_pdb_for_analysis(pdb_file: str, output_dir: str) -> Dict[str, str]:
    """Prepare existing PDB file for analysis."""
    antibody = {
        "name": Path(pdb_file).stem,
        "pdb_file": pdb_file,
        "type": "pdb"
    }
    
    config = {
        "paths": {"temp_dir": output_dir}
    }
    
    return prepare_structure(antibody, config)
