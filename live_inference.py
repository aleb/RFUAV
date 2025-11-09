import sys
import time
from pathlib import Path
from utils.benchmark import Classify_Model
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from threading import Thread
import time
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt

VALID_EXTENSIONS = {'.wav', '.iq'}

def main():
    cfg = sys.argv[1]
    weight_path = sys.argv[2]
    source = Path(sys.argv[3])

    test = Classify_Model(cfg, weight_path)
    test.start_visualization()

    if not source.is_dir():
        raise ValueError(f"{source} is not a valid directory")

    print(f"Monitoring {source} for new files...")

    try:
        while True:
            # Get all valid files that are younger than 1 second
            files_to_process = [
                f for f in source.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]

            if files_to_process:
                # Process each file immediately
                for file_path in files_to_process:
                    print(f"Processing {file_path}")
                    try:
                        test.liveGpuInference(str(file_path))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                    finally:
                        try:
#                            file_path.unlink()
                            print(f"Deleted {file_path}")
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
            else:
                # Sleep only if no new files available
                time.sleep(0.1)
                print(f"IDLE")

    except KeyboardInterrupt:
        test.terminate_visualization()

if __name__ == "__main__":
    main()
