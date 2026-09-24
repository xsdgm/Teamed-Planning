"""
Performance Timing Utilities for Multi-Robot System
"""

import time
from typing import Dict, Optional
from contextlib import contextmanager


class PerformanceTimer:
    """
    Performance timer for tracking execution time of different modules.
    
    Usage:
        timer = PerformanceTimer(robot_id="drone_0")
        
        with timer.measure("perception"):
            # ... perception code ...
            pass
        
        timer.print_stats()  # Print timing statistics
    """
    
    def __init__(self, robot_id: str):
        """
        Initialize performance timer.
        
        Args:
            robot_id: Identifier for this robot (drone/ground robot)
        """
        self.robot_id = robot_id
        self.timings = {}  # {module_name: [list of times]}
        self._current_start = None
        
    @contextmanager
    def measure(self, module_name: str):
        """
        Context manager for measuring execution time.
        
        Args:
            module_name: Name of the module being timed
            
        Usage:
            with timer.measure("semantic_processing"):
                # code to measure
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.record(module_name, elapsed)
    
    def record(self, module_name: str, elapsed_time: float):
        """
        Record timing for a module.
        
        Args:
            module_name: Name of the module
            elapsed_time: Elapsed time in seconds
        """
        if module_name not in self.timings:
            self.timings[module_name] = []
        self.timings[module_name].append(elapsed_time)
    
    def get_stats(self, module_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary with 'last', 'avg', 'min', 'max', 'total', 'count'
            or None if module has no recordings
        """
        if module_name not in self.timings or len(self.timings[module_name]) == 0:
            return None
        
        times = self.timings[module_name]
        return {
            'last': times[-1],
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times),
            'count': len(times)
        }
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in appropriate units (ms or s).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string with units
        """
        if seconds < 1.0:
            return f"{seconds * 1000:.1f} ms"
        else:
            return f"{seconds:.2f} s"
    
    def print_stats(self, summary: bool = False):
        """
        Print timing statistics to console.
        
        Args:
            summary: If True, print one-line summary. If False, print detailed stats.
        """
        if len(self.timings) == 0:
            return
        
        if summary:
            # One-line summary format
            parts = [f"[{self.robot_id}]"]
            for module_name, times in self.timings.items():
                last_time = times[-1]
                parts.append(f"{module_name}: {self.format_time(last_time)}")
            print(" | ".join(parts))
        else:
            # Detailed format
            print(f"\n{'='*60}")
            print(f"Performance Stats for {self.robot_id}")
            print(f"{'='*60}")
            
            for module_name in sorted(self.timings.keys()):
                stats = self.get_stats(module_name)
                if stats:
                    print(f"\n{module_name}:")
                    print(f"  Last:    {self.format_time(stats['last'])}")
                    print(f"  Average: {self.format_time(stats['avg'])}")
                    print(f"  Min/Max: {self.format_time(stats['min'])} / {self.format_time(stats['max'])}")
                    print(f"  Total:   {self.format_time(stats['total'])} ({stats['count']} calls)")
            
            print(f"{'='*60}\n")
    
    def reset(self):
        """Reset all timing data."""
        self.timings = {}
    
    def get_summary_string(self) -> str:
        """
        Get one-line summary string.
        
        Returns:
            Summary string with latest timings
        """
        if len(self.timings) == 0:
            return f"[{self.robot_id}] No timing data"
        
        parts = [f"[{self.robot_id}]"]
        for module_name, times in self.timings.items():
            last_time = times[-1]
            parts.append(f"{module_name}: {self.format_time(last_time)}")
        
        return " | ".join(parts)

