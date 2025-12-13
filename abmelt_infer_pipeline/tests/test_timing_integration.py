#!/usr/bin/env python3

"""
Standalone integration test for timing module.
Demonstrates the full timing flow without heavy pipeline dependencies.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timing import (
    TimingReport, 
    get_timing_report, 
    reset_timing_report, 
    time_step
)


def simulate_pipeline():
    """Simulate the AbMelt pipeline with timing."""
    print("=" * 60)
    print("SIMULATED PIPELINE WITH TIMING")
    print("=" * 60)
    
    # Initialize timing (like in infer.py main())
    reset_timing_report()
    timing_report = get_timing_report()
    timing_report.start()
    
    # Simulate Step 1: Structure Preparation
    with time_step("Structure Preparation"):
        print("Step 1: Simulating structure preparation...")
        time.sleep(0.2)
    
    # Simulate Step 2: MD Simulation with sub-steps
    with time_step("MD Simulation"):
        print("Step 2: Simulating MD simulation...")
        
        with time_step("GROMACS Preprocessing", parent="MD Simulation"):
            time.sleep(0.1)
        
        with time_step("System Setup", parent="MD Simulation"):
            time.sleep(0.1)
        
        with time_step("Multi-Temp Simulations", parent="MD Simulation"):
            time.sleep(0.3)
        
        with time_step("Trajectory Processing", parent="MD Simulation"):
            time.sleep(0.1)
    
    # Simulate Step 3: Descriptor Computation with sub-steps
    with time_step("Descriptor Computation"):
        print("Step 3: Simulating descriptor computation...")
        
        with time_step("GROMACS Descriptors", parent="Descriptor Computation"):
            time.sleep(0.1)
        
        with time_step("Order Parameters", parent="Descriptor Computation"):
            time.sleep(0.1)
        
        with time_step("Core/Surface SASA", parent="Descriptor Computation"):
            time.sleep(0.05)
        
        with time_step("Lambda Features", parent="Descriptor Computation"):
            time.sleep(0.05)
        
        with time_step("Aggregate to DataFrame", parent="Descriptor Computation"):
            time.sleep(0.05)
    
    # Simulate Step 4: Model Inference
    with time_step("Model Inference"):
        print("Step 4: Simulating model inference...")
        time.sleep(0.1)
    
    # Stop timing
    timing_report.stop()
    
    # Print summary (like at end of infer.py main())
    print(timing_report.format_summary())
    
    # Save to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "timing_test.json")
        timing_report.save_json(json_path)
        
        # Verify JSON was created correctly
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"\nJSON report saved successfully!")
        print(f"  Total entries: {len(data['entries'])}")
        print(f"  Total time: {data['total_duration_formatted']}")
    
    return timing_report


def main():
    """Run the integration test."""
    print("\n" + "=" * 60)
    print("TIMING MODULE INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    report = simulate_pipeline()
    
    # Verify structure
    expected_steps = [
        "Structure Preparation",
        "MD Simulation",
        "GROMACS Preprocessing",
        "System Setup",
        "Multi-Temp Simulations",
        "Trajectory Processing",
        "Descriptor Computation",
        "GROMACS Descriptors",
        "Order Parameters",
        "Core/Surface SASA",
        "Lambda Features",
        "Aggregate to DataFrame",
        "Model Inference"
    ]
    
    all_found = True
    for step in expected_steps:
        if step not in report.entries:
            print(f"✗ Missing step: {step}")
            all_found = False
    
    if all_found:
        print(f"\n✓ All {len(expected_steps)} timing entries present!")
        print("✓ Integration test PASSED!")
        return 0
    else:
        print("\n✗ Integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
