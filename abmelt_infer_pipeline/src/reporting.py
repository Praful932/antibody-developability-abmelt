import json
import time
import platform
import logging
from pathlib import Path

def save_benchmark_report(antibody, config, simulation_result):
    """
    Save MD simulation benchmark report.
    """
    try:
        report_data = {
            "antibody": antibody['name'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "simulation_config": {
                "temperatures": config['simulation']['temperatures'],
                "simulation_time": config['simulation']['simulation_time'],
                "gpu_enabled": config['simulation']['gpu_enabled'],
                "n_threads": config['gromacs']['n_threads']
            },
            "timings": simulation_result.get('timings', {})
        }

        # Try to get GPU info if possible (basic check)
        try:
            import subprocess
            nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_smi.returncode == 0:
                report_data["system_info"]["gpu_info"] = nvidia_smi.stdout.strip()
            else:
                report_data["system_info"]["gpu_info"] = "nvidia-smi failed or not found"
        except Exception:
             report_data["system_info"]["gpu_info"] = "Could not query GPU"

        output_dir = Path(config['paths']['output_dir'])
        report_file = output_dir / f"{antibody['name']}_benchmark.json"

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logging.info(f"Benchmark report saved to: {report_file}")

    except Exception as e:
        logging.error(f"Failed to save benchmark report: {e}")
