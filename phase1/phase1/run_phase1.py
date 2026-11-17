"""Run the five short stages in order with the current Python interpreter."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
for script in ["01_prepare.py", "02_network.py", "03_similarity.py", "04_checks.py", "05_report.py"]:
    print(f"\nRunning {script}", flush=True)
    subprocess.run([sys.executable, str(root / "code" / script)], env=env, check=True)
print("\nFinished. Open report/phase1_report.html")
