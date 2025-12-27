import os
import sys
import site
from pathlib import Path

print(f"Python: {sys.executable}")
lib_dirs = []
for p in sys.path:
    nvidia_path = Path(p) / "nvidia"
    if nvidia_path.exists():
        print(f"Nvidia path found: {nvidia_path}")
        libs = [str(d / "lib") for d in nvidia_path.iterdir() if (d / "lib").exists()]
        lib_dirs.extend(libs)

if lib_dirs:
    print(f"Total Lib dirs found: {len(lib_dirs)}")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
    os.environ["LD_LIBRARY_PATH"] = new_ld
    print(f"New LD_LIBRARY_PATH set in os.environ")
else:
    print("Nvidia path NOT found in any sys.path")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {gpus}")

from tensorflow.keras import mixed_precision
print(f"Mixed precision policy: {mixed_precision.global_policy()}")
