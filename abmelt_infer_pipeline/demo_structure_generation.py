#!/usr/bin/env python3

"""
Demo script showing how to use AbMelt structure generation functionality.
This script demonstrates the key features and usage patterns.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def demo_sequence_based_generation():
    """Demonstrate structure generation from sequences."""
    logger.info("=" * 60)
    logger.info("DEMO: Sequence-based Structure Generation")
    logger.info("=" * 60)

    try:
        from structure_prep import generate_structure_from_sequences, prepare_structure

        # Example antibody sequences (Alemtuzumab)
        heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS"
        light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR"

        logger.info("1. Direct structure generation using ImmuneBuilder...")
        output_file = "demo_alemtuzumab.pdb"
        generated_file = generate_structure_from_sequences(
            heavy_chain=heavy_chain, light_chain=light_chain, output_file=output_file
        )

        if Path(generated_file).exists():
            logger.info(f"✓ Structure generated: {generated_file}")
            logger.info(f"  File size: {Path(generated_file).stat().st_size} bytes")
        else:
            logger.error("✗ Structure generation failed")
            return False

        logger.info("\n2. Using prepare_structure function...")
        config = {
            "paths": {
                "temp_dir": "demo_temp",
                "output_dir": "demo_output",
                "log_dir": "demo_logs",
            }
        }

        # Create directories
        for path in config["paths"].values():
            Path(path).mkdir(parents=True, exist_ok=True)

        antibody = {
            "name": "alemtuzumab_demo",
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "type": "sequences",
        }

        structure_files = prepare_structure(antibody, config)

        logger.info("✓ Structure prepared successfully")
        logger.info(f"  PDB file: {structure_files['pdb_file']}")
        logger.info(f"  Work directory: {structure_files['work_dir']}")
        logger.info(f"  Chains found: {list(structure_files['chains'].keys())}")

        return True

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False


def demo_pdb_processing():
    """Demonstrate PDB file processing."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: PDB File Processing")
    logger.info("=" * 60)

    try:
        from structure_prep import (
            get_chain_sequences,
            prepare_pdb_for_analysis,
            validate_structure,
        )

        # Use the generated PDB file from previous demo
        pdb_file = "demo_alemtuzumab.pdb"

        if not Path(pdb_file).exists():
            logger.error(f"PDB file not found: {pdb_file}")
            logger.info("Please run the sequence generation demo first")
            return False

        logger.info(f"1. Processing existing PDB file: {pdb_file}")

        # Validate structure
        is_valid = validate_structure(pdb_file)
        logger.info(
            f"   Structure validation: {'✓ Valid' if is_valid else '✗ Invalid'}"
        )

        # Extract chain sequences
        chains = get_chain_sequences(pdb_file)
        logger.info(f"   Chains found: {list(chains.keys())}")

        for chain_id, sequence in chains.items():
            logger.info(f"   Chain {chain_id}: {len(sequence)} residues")
            logger.info(f"   First 20 residues: {sequence[:20]}...")

        logger.info("\n2. Using prepare_pdb_for_analysis...")
        structure_files = prepare_pdb_for_analysis(pdb_file, "demo_pdb_analysis")

        logger.info("✓ PDB processing completed")
        logger.info(f"  Processed PDB: {structure_files['pdb_file']}")
        logger.info(f"  Work directory: {structure_files['work_dir']}")

        return True

    except Exception as e:
        logger.error(f"PDB processing demo failed: {e}")
        return False


def demo_structure_analysis():
    """Demonstrate structure analysis capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Structure Analysis")
    logger.info("=" * 60)

    try:
        from Bio.PDB import PDBParser

        from structure_prep import get_chain_sequences

        pdb_file = "demo_alemtuzumab.pdb"

        if not Path(pdb_file).exists():
            logger.error(f"PDB file not found: {pdb_file}")
            return False

        logger.info(f"Analyzing structure: {pdb_file}")

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("antibody", pdb_file)

        # Basic structure information
        logger.info(f"Structure ID: {structure.id}")
        logger.info(f"Number of models: {len(list(structure.get_models()))}")

        # Chain information
        chains = list(structure.get_chains())
        logger.info(f"Number of chains: {len(chains)}")

        for _i, chain in enumerate(chains):
            residues = list(chain.get_residues())
            atoms = list(chain.get_atoms())
            logger.info(
                f"  Chain {chain.id}: {len(residues)} residues, {len(atoms)} atoms"
            )

            # Show first few residues
            if residues:
                first_residue = residues[0]
                logger.info(
                    f"    First residue: {first_residue.resname} {first_residue.id}"
                )

        # Sequence analysis
        sequences = get_chain_sequences(pdb_file)
        logger.info("\nSequence analysis:")
        for chain_id, sequence in sequences.items():
            logger.info(f"  Chain {chain_id}: {len(sequence)} amino acids")

            # Count amino acid types
            aa_counts = {}
            for aa in sequence:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1

            # Show most common amino acids
            common_aas = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"    Most common AAs: {common_aas}")

        return True

    except Exception as e:
        logger.error(f"Structure analysis demo failed: {e}")
        return False


def cleanup_demo_files():
    """Clean up demo files."""
    logger.info("\n" + "=" * 60)
    logger.info("CLEANUP")
    logger.info("=" * 60)

    import shutil

    files_to_remove = ["demo_alemtuzumab.pdb"]

    dirs_to_remove = ["demo_temp", "demo_output", "demo_logs", "demo_pdb_analysis"]

    # Remove files
    for file_path in files_to_remove:
        if Path(file_path).exists():
            Path(file_path).unlink()
            logger.info(f"Removed file: {file_path}")

    # Remove directories
    for dir_path in dirs_to_remove:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            logger.info(f"Removed directory: {dir_path}")

    logger.info("✓ Cleanup completed")


def main():
    """Run the complete demo."""
    logger.info("ABMELT STRUCTURE GENERATION DEMO")
    logger.info("This demo shows how to use the structure generation functionality")

    try:
        # Run demos
        success = True

        success &= demo_sequence_based_generation()
        success &= demo_pdb_processing()
        success &= demo_structure_analysis()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 60)

        if success:
            logger.info("🎉 All demos completed successfully!")
            logger.info("\nKey takeaways:")
            logger.info(
                "• Use generate_structure_from_sequences() for direct generation"
            )
            logger.info("• Use prepare_structure() for full pipeline integration")
            logger.info("• Use prepare_pdb_for_analysis() for existing PDB files")
            logger.info("• Structure validation and sequence extraction are automatic")
        else:
            logger.error("❌ Some demos failed")

        # Cleanup
        cleanup_demo_files()

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
        cleanup_demo_files()
        return 1
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        cleanup_demo_files()
        return 1


if __name__ == "__main__":
    sys.exit(main())
