
import subprocess
import sys

try:
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
