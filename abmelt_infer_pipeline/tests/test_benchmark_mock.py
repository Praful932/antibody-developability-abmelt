import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
from pathlib import Path

# Add src directories to path
current_dir = Path(__file__).parent.resolve()
src_dir = current_dir.parent / "src"
pipeline_dir = current_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(pipeline_dir))

# Mock modules BEFORE importing md_simulation
sys.modules["gromacs"] = MagicMock()
sys.modules["gromacs.config"] = MagicMock()
sys.modules["gromacs.tools"] = MagicMock()

# Mock preprocess module
mock_preprocess = MagicMock()
mock_preprocess.protonation_state = MagicMock(return_value=["1", "1"])
mock_preprocess.canonical_index = MagicMock(return_value=["index_group"])
mock_preprocess.edit_mdp = MagicMock()
sys.modules["preprocess"] = mock_preprocess

# Now import modules under test
import md_simulation
from reporting import save_benchmark_report

class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            "simulation": {
                "temperatures": [300, 350],
                "simulation_time": 0.002,
                "pH": 7.4,
                "force_field": "charmm27",
                "water_model": "tip3p",
                "salt_concentration": 150,
                "p_salt": "NA",
                "n_salt": "CL",
                "gpu_enabled": False
            },
            "gromacs": {
                "mdp_dir": str(self.test_dir / "mdp"),
                "n_threads": 1,
                "gpu_id": 0
            },
            "paths": {
                "temp_dir": self.test_dir / "temp",
                "output_dir": self.test_dir / "output"
            }
        }

        # Create directories
        (self.test_dir / "mdp").mkdir()
        (self.test_dir / "temp").mkdir()
        (self.test_dir / "output").mkdir()

        # Create dummy files
        (self.test_dir / "test.pdb").touch()

        # Setup mocks for gromacs
        md_simulation.gromacs.config.templates = {}
        # Make get_templates return a list with a path to existing file
        md_simulation.gromacs.config.get_templates = MagicMock(return_value=[str(self.test_dir / "template.mdp")])
        md_simulation.gromacs.config.path = [str(self.test_dir / "mdp")]
        (self.test_dir / "template.mdp").touch()

        # Mock file operations inside md_simulation if needed
        # We also need to ensure that mdp files are "found" or "created"
        # The code checks `if src_file.exists()` in `run_md_simulation`
        # We can just create them in setUp
        for mdp in ["nvt.mdp", "npt.mdp", "md.mdp", "ions.mdp", "em.mdp"]:
             (self.test_dir / "mdp" / mdp).touch()

        # Create dummy tpr/xtc files expected by process_trajectories for custom simulation time
        # simulation_time is 0.002
        for temp in [300, 350]:
            (self.test_dir / f"md_{temp}_0.002.tpr").touch()
            (self.test_dir / f"md_{temp}_0.002.xtc").touch()


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_md_simulation_timings(self):
        # Setup mocks
        structure_files = {
            "pdb_file": str(self.test_dir / "test.pdb"),
            "work_dir": str(self.test_dir)
        }

        # Run MD simulation
        result = md_simulation.run_md_simulation(structure_files, self.config)

        # Check result
        self.assertIn("timings", result)
        self.assertIn("preprocessing", result["timings"])
        self.assertIn("system_setup", result["timings"])
        self.assertIn("simulation_total", result["timings"])
        self.assertIn("simulation_details", result["timings"])
        self.assertIn("postprocessing", result["timings"])

        details = result["timings"]["simulation_details"]
        self.assertIn("300", details)
        self.assertIn("350", details)

        self.assertIn("nvt", details["300"])
        self.assertIn("npt", details["300"])
        self.assertIn("md", details["300"])

        print("\nTest passed. Timings captured:")
        print(result["timings"])

        # Test report saving
        antibody = {"name": "test_antibody"}
        save_benchmark_report(antibody, self.config, result)

        report_file = self.test_dir / "output" / "test_antibody_benchmark.json"
        self.assertTrue(report_file.exists())

        import json
        with open(report_file) as f:
            report = json.load(f)
            self.assertIn("timings", report)
            self.assertIn("system_info", report)
            print("Report content:", json.dumps(report, indent=2))

if __name__ == '__main__':
    unittest.main()
