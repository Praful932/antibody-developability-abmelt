#!/usr/bin/env python3

"""
Timing utilities for AbMelt inference pipeline.
Provides context managers and report generation for measuring pipeline performance.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TimingEntry:
    """Single timing entry with start/end times."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    parent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "duration_seconds": self.duration,
            "parent": self.parent
        }


class TimingReport:
    """
    Aggregates timing data with hierarchical structure.
    
    Usage:
        report = TimingReport()
        with report.time("Step 1"):
            # do work
            with report.time("Sub-step 1.1", parent="Step 1"):
                # do sub-work
        
        print(report.format_summary())
        report.save_json("timing.json")
    """
    
    def __init__(self):
        self.entries: Dict[str, TimingEntry] = {}
        self.order: List[str] = []  # Preserve insertion order
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        
    def start(self):
        """Mark the start of the overall timing."""
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Mark the end of the overall timing."""
        self.end_time = time.perf_counter()
    
    @property
    def total_duration(self) -> float:
        """Total pipeline duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time
    
    @contextmanager
    def time(self, name: str, parent: Optional[str] = None):
        """
        Context manager for timing a code block.
        
        Args:
            name: Name of the step being timed
            parent: Optional parent step name for hierarchical display
        """
        entry = TimingEntry(name=name, parent=parent)
        entry.start_time = time.perf_counter()
        
        try:
            yield entry
        finally:
            entry.end_time = time.perf_counter()
            entry.duration = entry.end_time - entry.start_time
            self.entries[name] = entry
            if name not in self.order:
                self.order.append(name)
            logger.info(f"[TIMING] {name}: {self._format_duration(entry.duration)}")
    
    def add_entry(self, name: str, duration: float, parent: Optional[str] = None):
        """Manually add a timing entry."""
        entry = TimingEntry(name=name, duration=duration, parent=parent)
        self.entries[name] = entry
        if name not in self.order:
            self.order.append(name)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.0f}s"
    
    def format_summary(self) -> str:
        """Generate human-readable timing summary."""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("PIPELINE TIMING REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Pipeline Time: {self._format_duration(self.total_duration)}")
        lines.append("")
        lines.append(f"{'Step':<45} {'Time':>10} {'%':>6}")
        lines.append("-" * 60)
        
        # Group by parent
        top_level = [name for name in self.order if self.entries[name].parent is None]
        
        for i, name in enumerate(top_level, 1):
            entry = self.entries[name]
            pct = (entry.duration / self.total_duration * 100) if self.total_duration > 0 else 0
            lines.append(f"{i}. {name:<42} {self._format_duration(entry.duration):>10} {pct:>5.1f}%")
            
            # Find children
            children = [n for n in self.order if self.entries[n].parent == name]
            for j, child_name in enumerate(children):
                child = self.entries[child_name]
                child_pct = (child.duration / self.total_duration * 100) if self.total_duration > 0 else 0
                prefix = "└─" if j == len(children) - 1 else "├─"
                lines.append(f"   {prefix} {child_name:<39} {self._format_duration(child.duration):>10} {child_pct:>5.1f}%")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_duration_seconds": self.total_duration,
            "total_duration_formatted": self._format_duration(self.total_duration),
            "entries": [self.entries[name].to_dict() for name in self.order]
        }
    
    def save_json(self, filepath: str):
        """Save timing report to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Timing report saved to: {filepath}")
    
    def save_csv(self, filepath: str):
        """Save timing report to CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("step,parent,duration_seconds,duration_formatted,percentage\n")
            for name in self.order:
                entry = self.entries[name]
                pct = (entry.duration / self.total_duration * 100) if self.total_duration > 0 else 0
                parent = entry.parent or ""
                f.write(f'"{name}","{parent}",{entry.duration:.3f},"{self._format_duration(entry.duration)}",{pct:.2f}\n')
        
        logger.info(f"Timing report saved to: {filepath}")


# Global timing report instance for pipeline use
_global_report: Optional[TimingReport] = None


def get_timing_report() -> TimingReport:
    """Get or create the global timing report instance."""
    global _global_report
    if _global_report is None:
        _global_report = TimingReport()
    return _global_report


def reset_timing_report():
    """Reset the global timing report."""
    global _global_report
    _global_report = None


@contextmanager
def time_step(name: str, parent: Optional[str] = None):
    """
    Convenience function to time a step using the global report.
    
    Args:
        name: Name of the step
        parent: Optional parent step name
    """
    report = get_timing_report()
    with report.time(name, parent=parent) as entry:
        yield entry
