#!/usr/bin/env python3

"""
Test runner script for AbMelt structure generation.
Provides an easy way to run different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_name, args=None):
    """Run a Python script and return the result."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_name}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run AbMelt structure generation tests')
    parser.add_argument('test_type', 
                       choices=['quick', 'simple', 'comprehensive', 'demo', 'all'],
                       help='Type of test to run')
    parser.add_argument('--keep-files', action='store_true',
                       help='Keep test files (for comprehensive tests)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ABMELT STRUCTURE GENERATION TEST RUNNER")
    print("=" * 60)
    
    success = True
    
    if args.test_type in ['quick', 'all']:
        print("\n1. Running Quick Test...")
        success &= run_script('quick_test.py')
    
    if args.test_type in ['simple', 'all']:
        print("\n2. Running Simple Test...")
        success &= run_script('simple_structure_test.py')
    
    if args.test_type in ['comprehensive', 'all']:
        print("\n3. Running Comprehensive Test...")
        test_args = []
        if args.keep_files:
            test_args.append('--keep-files')
        if args.verbose:
            test_args.append('--verbose')
        success &= run_script('test_structure_generation.py', test_args)
    
    if args.test_type in ['demo', 'all']:
        print("\n4. Running Demo...")
        success &= run_script('demo_structure_generation.py')
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("• Test MD simulation: Check md_simulation.py")
        print("• Test full pipeline: Run infer.py")
        print("• Check generated structures in test directories")
    else:
        print("❌ Some tests failed")
        print("\nTroubleshooting:")
        print("• Check dependencies: pip install biopython ImmuneBuilder")
        print("• Verify environment setup")
        print("• Check file permissions")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
