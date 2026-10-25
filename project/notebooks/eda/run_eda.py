import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("01_catalog.py", "02_attributes.py", "03_search_priority.py"):
    print(f"\n{'=' * 60}\nЗапуск: {script}\n{'=' * 60}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
print("\nГотово. Результаты: project/EDA/reports/")
