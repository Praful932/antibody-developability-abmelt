#!/usr/bin/env python3

"""
Test script for AbMelt structure generation functionality.
Tests both sequence-based structure generation and PDB-based processing.
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, cast

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from Bio.PDB import PDBParser

    from structure_prep import (
        generate_structure_from_sequences,
        get_chain_sequences,
        prepare_pdb_for_analysis,
        prepare_structure,
        validate_structure,
    )
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print(
        "Please ensure you're in the correct environment with required dependencies installed."
    )
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class StructureGenerationTester:
    """Test class for structure generation functionality."""

    def __init__(self, test_dir: Optional[str] = None):
        """Initialize tester with optional test directory."""
        self.test_dir = (
            Path(test_dir)
            if test_dir
            else Path(tempfile.mkdtemp(prefix="abmelt_test_"))
        )
        self.test_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Test directory: {self.test_dir}")

        # Test antibody sequences (example sequences)
        self.test_sequences = {
            "alemtuzumab": {
                "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS",
                "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR",
            },
            "nivolumab": {
                "heavy": "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYWINWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS",
                "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR",
            },
        }

        # Test configuration
        self.config = {
            "paths": {
                "temp_dir": str(self.test_dir),
                "output_dir": str(self.test_dir / "output"),
                "log_dir": str(self.test_dir / "logs"),
            },
            "structure": {
                "validate_structure": True,
                "extract_sequences": True,
                "create_work_dir": True,
            },
        }

        # Create output directories
        for path_str in cast(dict[str, str], self.config["paths"]).values():
            Path(path_str).mkdir(parents=True, exist_ok=True)

    def test_sequence_based_generation(self) -> dict[str, bool]:
        """Test structure generation from sequences."""
        logger.info("=" * 60)
        logger.info("TESTING SEQUENCE-BASED STRUCTURE GENERATION")
        logger.info("=" * 60)

        results = {}

        for antibody_name, sequences in self.test_sequences.items():
            logger.info(f"\nTesting {antibody_name}...")

            try:
                # Test direct sequence generation
                output_file = self.test_dir / f"{antibody_name}_direct.pdb"
                logger.info("Testing direct sequence generation...")

                generated_file = generate_structure_from_sequences(
                    heavy_chain=sequences["heavy"],
                    light_chain=sequences["light"],
                    output_file=str(output_file),
                )

                # Verify file was created
                if Path(generated_file).exists():
                    logger.info(f"✓ Direct generation successful: {generated_file}")
                    results[f"{antibody_name}_direct"] = True
                else:
                    logger.error(f"✗ Direct generation failed: {generated_file}")
                    results[f"{antibody_name}_direct"] = False

                # Test through prepare_structure function
                logger.info("Testing through prepare_structure...")
                antibody = {
                    "name": antibody_name,
                    "heavy_chain": sequences["heavy"],
                    "light_chain": sequences["light"],
                    "type": "sequences",
                }

                structure_files = prepare_structure(antibody, self.config)

                # Verify structure files
                if self._verify_structure_files(structure_files, antibody_name):
                    logger.info(f"✓ prepare_structure successful for {antibody_name}")
                    results[f"{antibody_name}_prepare"] = True
                else:
                    logger.error(f"✗ prepare_structure failed for {antibody_name}")
                    results[f"{antibody_name}_prepare"] = False

            except Exception as e:
                logger.error(f"✗ Error testing {antibody_name}: {e}")
                results[f"{antibody_name}_error"] = False

        return results

    def test_pdb_based_processing(self) -> dict[str, bool]:
        """Test PDB-based structure processing."""
        logger.info("=" * 60)
        logger.info("TESTING PDB-BASED STRUCTURE PROCESSING")
        logger.info("=" * 60)

        results = {}

        # First generate some test PDBs
        test_pdbs: dict[str, Path] = {}
        for antibody_name, sequences in self.test_sequences.items():
            try:
                pdb_file_path = self.test_dir / f"{antibody_name}_test.pdb"
                generate_structure_from_sequences(
                    heavy_chain=sequences["heavy"],
                    light_chain=sequences["light"],
                    output_file=str(pdb_file_path),
                )
                test_pdbs[antibody_name] = pdb_file_path
                logger.info(f"Generated test PDB: {pdb_file_path}")
            except Exception as e:
                logger.error(f"Failed to generate test PDB for {antibody_name}: {e}")
                continue

        # Test PDB processing
        for antibody_name, pdb_file_path in test_pdbs.items():
            logger.info(f"\nTesting PDB processing for {antibody_name}...")

            try:
                # Test prepare_pdb_for_analysis
                logger.info("Testing prepare_pdb_for_analysis...")
                structure_files = prepare_pdb_for_analysis(
                    pdb_file=str(pdb_file_path), output_dir=str(self.test_dir / "pdb_analysis")
                )

                if self._verify_structure_files(structure_files, antibody_name):
                    logger.info(
                        f"✓ prepare_pdb_for_analysis successful for {antibody_name}"
                    )
                    results[f"{antibody_name}_pdb_analysis"] = True
                else:
                    logger.error(
                        f"✗ prepare_pdb_for_analysis failed for {antibody_name}"
                    )
                    results[f"{antibody_name}_pdb_analysis"] = False

                # Test through prepare_structure with PDB type
                logger.info("Testing prepare_structure with PDB type...")
                antibody = {
                    "name": f"{antibody_name}_pdb",
                    "pdb_file": str(pdb_file_path),
                    "type": "pdb",
                }

                structure_files = prepare_structure(antibody, self.config)

                if self._verify_structure_files(
                    structure_files, f"{antibody_name}_pdb"
                ):
                    logger.info(
                        f"✓ prepare_structure (PDB) successful for {antibody_name}"
                    )
                    results[f"{antibody_name}_pdb_prepare"] = True
                else:
                    logger.error(
                        f"✗ prepare_structure (PDB) failed for {antibody_name}"
                    )
                    results[f"{antibody_name}_pdb_prepare"] = False

            except Exception as e:
                logger.error(f"✗ Error processing PDB for {antibody_name}: {e}")
                results[f"{antibody_name}_pdb_error"] = False

        return results

    def test_structure_validation(self) -> dict[str, bool]:
        """Test structure validation functionality."""
        logger.info("=" * 60)
        logger.info("TESTING STRUCTURE VALIDATION")
        logger.info("=" * 60)

        results = {}

        # Test with valid structures
        for antibody_name, sequences in self.test_sequences.items():
            try:
                pdb_file = self.test_dir / f"{antibody_name}_validation.pdb"
                generate_structure_from_sequences(
                    heavy_chain=sequences["heavy"],
                    light_chain=sequences["light"],
                    output_file=str(pdb_file),
                )

                # Test validation
                is_valid = validate_structure(str(pdb_file))
                if is_valid:
                    logger.info(f"✓ Structure validation passed for {antibody_name}")
                    results[f"{antibody_name}_validation"] = True
                else:
                    logger.warning(f"⚠ Structure validation failed for {antibody_name}")
                    results[f"{antibody_name}_validation"] = False

                # Test sequence extraction
                chains = get_chain_sequences(str(pdb_file))
                if chains:
                    logger.info(
                        f"✓ Chain sequences extracted for {antibody_name}: {list(chains.keys())}"
                    )
                    results[f"{antibody_name}_sequences"] = True
                else:
                    logger.error(
                        f"✗ Failed to extract chain sequences for {antibody_name}"
                    )
                    results[f"{antibody_name}_sequences"] = False

            except Exception as e:
                logger.error(f"✗ Error in validation test for {antibody_name}: {e}")
                results[f"{antibody_name}_validation_error"] = False

        # Test with invalid file
        try:
            invalid_file = self.test_dir / "invalid.pdb"
            with open(invalid_file, "w") as f:
                f.write(
                    "ATOM      1  N   ALA A   1      20.154  16.967  23.862  1.00 11.18           N\n"
                )
                f.write(
                    "ATOM      2  CA  ALA A   1      19.030  16.067  23.862  1.00 11.18           C\n"
                )
                # Incomplete structure - only one chain

            is_valid = validate_structure(str(invalid_file))
            if not is_valid:
                logger.info("✓ Correctly identified invalid structure (single chain)")
                results["invalid_structure"] = True
            else:
                logger.warning("⚠ Failed to identify invalid structure")
                results["invalid_structure"] = False

        except Exception as e:
            logger.error(f"✗ Error testing invalid structure: {e}")
            results["invalid_structure_error"] = False

        return results

    def test_error_handling(self) -> dict[str, bool]:
        """Test error handling for various edge cases."""
        logger.info("=" * 60)
        logger.info("TESTING ERROR HANDLING")
        logger.info("=" * 60)

        results = {}

        # Test with invalid antibody type
        try:
            invalid_antibody = {"name": "test", "type": "invalid_type"}
            prepare_structure(invalid_antibody, self.config)
            logger.error("✗ Should have raised error for invalid antibody type")
            results["invalid_type"] = False
        except ValueError as e:
            logger.info(f"✓ Correctly raised error for invalid type: {e}")
            results["invalid_type"] = True
        except Exception as e:
            logger.error(f"✗ Unexpected error for invalid type: {e}")
            results["invalid_type"] = False

        # Test with missing sequences
        try:
            incomplete_antibody = {
                "name": "test",
                "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS",
                "type": "sequences",
                # Missing light_chain
            }
            prepare_structure(incomplete_antibody, self.config)
            logger.error("✗ Should have raised error for missing light chain")
            results["missing_light"] = False
        except KeyError as e:
            logger.info(f"✓ Correctly raised error for missing light chain: {e}")
            results["missing_light"] = True
        except Exception as e:
            logger.error(f"✗ Unexpected error for missing light chain: {e}")
            results["missing_light"] = False

        # Test with non-existent PDB file
        try:
            non_existent_antibody = {
                "name": "test",
                "pdb_file": "/non/existent/file.pdb",
                "type": "pdb",
            }
            prepare_structure(non_existent_antibody, self.config)
            logger.error("✗ Should have raised error for non-existent PDB")
            results["non_existent_pdb"] = False
        except FileNotFoundError as e:
            logger.info(f"✓ Correctly raised error for non-existent PDB: {e}")
            results["non_existent_pdb"] = True
        except Exception as e:
            logger.error(f"✗ Unexpected error for non-existent PDB: {e}")
            results["non_existent_pdb"] = False

        return results

    def _verify_structure_files(
        self, structure_files: dict[str, str], antibody_name: str
    ) -> bool:
        """Verify that structure files were created correctly."""
        required_keys = ["pdb_file", "work_dir"]

        # Check required keys
        for key in required_keys:
            if key not in structure_files:
                logger.error(f"Missing required key: {key}")
                return False

        # Check if files exist
        pdb_file = Path(structure_files["pdb_file"])
        work_dir = Path(structure_files["work_dir"])

        if not pdb_file.exists():
            logger.error(f"PDB file does not exist: {pdb_file}")
            return False

        if not work_dir.exists():
            logger.error(f"Work directory does not exist: {work_dir}")
            return False

        # Check if PDB file is valid
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("test", str(pdb_file))
            chains = list(structure.get_chains())
            if len(chains) < 2:
                logger.error(f"PDB file has insufficient chains: {len(chains)}")
                return False
        except Exception as e:
            logger.error(f"PDB file is not valid: {e}")
            return False

        logger.info(f"✓ Structure files verified for {antibody_name}")
        return True

    def run_all_tests(self) -> dict[str, bool]:
        """Run all tests and return combined results."""
        logger.info("Starting comprehensive structure generation tests...")

        all_results = {}

        # Run all test categories
        all_results.update(self.test_sequence_based_generation())
        all_results.update(self.test_pdb_based_processing())
        all_results.update(self.test_structure_validation())
        all_results.update(self.test_error_handling())

        return all_results

    def print_summary(self, results: dict[str, bool]) -> None:
        """Print test summary."""
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        failed_tests = total_tests - passed_tests

        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

        if failed_tests > 0:
            logger.info("\nFailed tests:")
            for test_name, result in results.items():
                if not result:
                    logger.info(f"  ✗ {test_name}")

        logger.info(f"\nTest directory: {self.test_dir}")
        logger.info("You can inspect the generated files in the test directory.")

    def cleanup(self) -> None:
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")


def main() -> int:
    """Main function to run structure generation tests."""
    parser = argparse.ArgumentParser(description="Test AbMelt structure generation")
    parser.add_argument(
        "--test-dir", type=str, help="Test directory (default: temporary)"
    )
    parser.add_argument(
        "--keep-files", action="store_true", help="Keep test files after completion"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create tester
    tester = StructureGenerationTester(test_dir=args.test_dir)

    try:
        # Run all tests
        results = tester.run_all_tests()

        # Print summary
        tester.print_summary(results)

        # Cleanup unless keeping files
        if not args.keep_files:
            tester.cleanup()
        else:
            logger.info(f"Test files kept in: {tester.test_dir}")

        # Exit with appropriate code
        failed_tests = sum(1 for result in results.values() if not result)
        sys.exit(0 if failed_tests == 0 else 1)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        tester.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
