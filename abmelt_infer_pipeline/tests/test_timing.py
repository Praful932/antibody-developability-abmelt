#!/usr/bin/env python3

"""
Unit tests for the timing module.
Tests TimingReport, TimingContext, and related utilities.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timing import (
    TimingReport, 
    TimingEntry, 
    get_timing_report, 
    reset_timing_report, 
    time_step
)


class TestTimingEntry:
    """Tests for TimingEntry dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = TimingEntry(name="test_step", duration=1.5, parent="parent_step")
        d = entry.to_dict()
        
        assert d["name"] == "test_step"
        assert d["duration_seconds"] == 1.5
        assert d["parent"] == "parent_step"


class TestTimingReport:
    """Tests for TimingReport class."""
    
    def test_basic_timing(self):
        """Test basic timing functionality."""
        report = TimingReport()
        report.start()
        
        with report.time("Step 1"):
            time.sleep(0.1)
        
        report.stop()
        
        assert "Step 1" in report.entries
        assert report.entries["Step 1"].duration >= 0.1
        assert report.total_duration >= 0.1
    
    def test_nested_timing(self):
        """Test hierarchical timing with parent."""
        report = TimingReport()
        report.start()
        
        with report.time("Parent Step"):
            time.sleep(0.05)
            with report.time("Child Step", parent="Parent Step"):
                time.sleep(0.05)
        
        report.stop()
        
        assert "Parent Step" in report.entries
        assert "Child Step" in report.entries
        assert report.entries["Child Step"].parent == "Parent Step"
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert TimingReport._format_duration(30) == "30.00s"
        assert TimingReport._format_duration(90) == "1m 30.0s"
        assert TimingReport._format_duration(3661) == "1h 1m 1s"
    
    def test_format_summary(self):
        """Test summary generation."""
        report = TimingReport()
        report.start()
        
        with report.time("Step 1"):
            time.sleep(0.01)
        
        with report.time("Step 2"):
            time.sleep(0.01)
        
        report.stop()
        
        summary = report.format_summary()
        
        assert "PIPELINE TIMING REPORT" in summary
        assert "Step 1" in summary
        assert "Step 2" in summary
        assert "Total Pipeline Time" in summary
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = TimingReport()
        report.start()
        
        with report.time("Step 1"):
            time.sleep(0.01)
        
        report.stop()
        
        d = report.to_dict()
        
        assert "total_duration_seconds" in d
        assert "total_duration_formatted" in d
        assert "entries" in d
        assert len(d["entries"]) == 1
        assert d["entries"][0]["name"] == "Step 1"
    
    def test_save_json(self):
        """Test saving to JSON file."""
        report = TimingReport()
        report.start()
        
        with report.time("Step 1"):
            time.sleep(0.01)
        
        report.stop()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "timing.json")
            report.save_json(filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "total_duration_seconds" in data
            assert len(data["entries"]) == 1
    
    def test_save_csv(self):
        """Test saving to CSV file."""
        report = TimingReport()
        report.start()
        
        with report.time("Step 1"):
            time.sleep(0.01)
        
        report.stop()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "timing.csv")
            report.save_csv(filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) >= 2  # Header + at least 1 data row
            assert "step,parent,duration_seconds" in lines[0]
    
    def test_add_entry_manually(self):
        """Test manually adding entries."""
        report = TimingReport()
        report.add_entry("Manual Step", duration=5.0, parent=None)
        
        assert "Manual Step" in report.entries
        assert report.entries["Manual Step"].duration == 5.0


class TestGlobalReport:
    """Tests for global timing report functions."""
    
    def test_get_timing_report(self):
        """Test getting global report."""
        reset_timing_report()
        report1 = get_timing_report()
        report2 = get_timing_report()
        
        assert report1 is report2
    
    def test_reset_timing_report(self):
        """Test resetting global report."""
        reset_timing_report()
        report1 = get_timing_report()
        
        reset_timing_report()
        report2 = get_timing_report()
        
        assert report1 is not report2
    
    def test_time_step_convenience(self):
        """Test time_step convenience function."""
        reset_timing_report()
        report = get_timing_report()
        
        with time_step("Convenience Step"):
            time.sleep(0.01)
        
        assert "Convenience Step" in report.entries


def run_tests():
    """Run all tests and print results."""
    import traceback
    
    test_classes = [TestTimingEntry, TestTimingReport, TestGlobalReport]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("TIMING MODULE UNIT TESTS")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
