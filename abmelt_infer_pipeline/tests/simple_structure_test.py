#!/usr/bin/env python3

"""
Simple test script for AbMelt structure generation.
Quick test to verify structure generation functionality works.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports (go up from tests/ to abmelt_infer_pipeline/, then into src/)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from structure_prep import (
        generate_structure_from_sequences,  # noqa: F401
        get_chain_sequences,
        prepare_structure,  # noqa: F401
        validate_structure,
    )

    logger.info("✓ Successfully imported structure_prep modules")
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Please ensure required dependencies are installed and PYTHONPATH is set.")
    sys.exit(1)


def test_imports() -> bool:
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    logger.info("✓ Successfully imported structure_prep modules")
    return True


def test_immune_builder() -> bool:
    """Test ImmuneBuilder functionality."""
    logger.info("Testing ImmuneBuilder...")

    try:
        from structure_prep import generate_structure_from_sequences

        # Test sequences (shortened for testing)
        heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS"
        light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR"

        # Generate structure
        output_file = "test_antibody.pdb"
        generated_file = generate_structure_from_sequences(
            heavy_chain=heavy_chain, light_chain=light_chain, output_file=output_file
        )

        # Check if file was created
        if Path(generated_file).exists():
            logger.info(f"✓ ImmuneBuilder generated structure: {generated_file}")

            # Test validation
            is_valid = validate_structure(generated_file)
            if is_valid:
                logger.info("✓ Structure validation passed")
            else:
                logger.warning("⚠ Structure validation failed")

            # Test sequence extraction
            chains = get_chain_sequences(generated_file)
            if chains:
                logger.info(f"✓ Extracted chains: {list(chains.keys())}")
            else:
                logger.warning("⚠ Failed to extract chain sequences")

            # Cleanup
            Path(generated_file).unlink()
            logger.info("✓ Cleaned up test file")

            return True
        else:
            logger.error("✗ ImmuneBuilder failed to generate structure")
            return False

    except Exception as e:
        logger.error(f"✗ ImmuneBuilder test failed: {e}")
        return False


def test_prepare_structure() -> bool:
    """Test the main prepare_structure function."""
    logger.info("Testing prepare_structure function...")

    try:
        from structure_prep import prepare_structure

        # Test configuration
        config = {
            "paths": {
                "temp_dir": "test_temp",
                "output_dir": "test_output",
                "log_dir": "test_logs",
            }
        }

        # Create test directories
        for path in config["paths"].values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Test antibody data
        antibody = {
            "name": "test_antibody",
            "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS",
            "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR",
            "type": "sequences",
        }

        # Run prepare_structure
        structure_files = prepare_structure(antibody, config)

        # Verify results
        required_keys = ["pdb_file", "work_dir", "chains"]
        for key in required_keys:
            if key not in structure_files:
                logger.error(f"✗ Missing required key: {key}")
                return False

        # Check if files exist
        pdb_file = Path(structure_files["pdb_file"])
        work_dir = Path(structure_files["work_dir"])

        if not pdb_file.exists():
            logger.error(f"✗ PDB file does not exist: {pdb_file}")
            return False

        if not work_dir.exists():
            logger.error(f"✗ Work directory does not exist: {work_dir}")
            return False

        logger.info("✓ prepare_structure successful")
        logger.info(f"  PDB file: {pdb_file}")
        logger.info(f"  Work dir: {work_dir}")
        logger.info(f"  Chains: {list(structure_files['chains'].keys())}")

        # Cleanup
        import shutil

        shutil.rmtree("test_temp")
        shutil.rmtree("test_output")
        shutil.rmtree("test_logs")
        logger.info("✓ Cleaned up test directories")

        return True

    except Exception as e:
        logger.error(f"✗ prepare_structure test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("ABMELT STRUCTURE GENERATION TEST")
    logger.info("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("ImmuneBuilder Test", test_immune_builder),
        ("Prepare Structure Test", test_prepare_structure),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
