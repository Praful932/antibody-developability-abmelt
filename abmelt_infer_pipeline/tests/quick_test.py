#!/usr/bin/env python3

"""
Quick test script for AbMelt structure generation.
Minimal test that can be run to verify basic functionality.
"""

import sys
from pathlib import Path


def test_imports() -> bool:
    """Test basic imports."""
    print("Testing imports...")

    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))

        # Test BioPython
        from Bio.PDB import PDBParser  # noqa: F401
        from Bio.SeqUtils import seq1  # noqa: F401

        print("✓ BioPython imports successful")

        # Test ImmuneBuilder
        from ImmuneBuilder import ABodyBuilder2  # noqa: F401

        print("✓ ImmuneBuilder import successful")

        # Test structure_prep
        from structure_prep import generate_structure_from_sequences  # noqa: F401

        print("✓ structure_prep import successful")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_generation() -> bool:
    """Test basic structure generation."""
    print("\nTesting basic structure generation...")

    try:
        from structure_prep import generate_structure_from_sequences

        # Short test sequences
        heavy = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATITADESTSTTAYMELSSLRSEDTAVYYCARGGYSSGYYFDYWGQGTLVTVSS"
        light = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWFQQKPGKAPKLLIYYATSLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTFPWTFGQGTKVEIKR"

        # Generate structure
        output_file = "quick_test.pdb"
        result = generate_structure_from_sequences(heavy, light, output_file)

        # Check if file was created
        if Path(result).exists():
            print(f"✓ Structure generated: {result}")
            print(f"  File size: {Path(result).stat().st_size} bytes")

            # Clean up
            Path(result).unlink()
            print("✓ Test file cleaned up")
            return True
        else:
            print("✗ Structure file not created")
            return False

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


def main() -> int:
    """Run quick tests."""
    print("=" * 50)
    print("ABMELT QUICK TEST")
    print("=" * 50)

    tests = [("Import Test", test_imports), ("Generation Test", test_basic_generation)]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        print("\nYou can now run the full test suite:")
        print("  python simple_structure_test.py")
        print("  python test_structure_generation.py")
        return 0
    else:
        print("❌ Some tests failed")
        print("\nPlease check your environment setup:")
        print("  pip install biopython")
        print("  pip install ImmuneBuilder")
        return 1


if __name__ == "__main__":
    sys.exit(main())
