import os
import csv

RES_ROOT = "./res"      # root folder containing all model results
DATASET_ROOT = "./dataset"  # folder with .iq files (dataset names from filenames)

# Get all dataset names
dataset_files = sorted(os.listdir(DATASET_ROOT))
dataset_names = [os.path.splitext(f)[0] for f in dataset_files if f.endswith(".iq")]

# Prepare CSV header
header = ["model"] + dataset_names

# Open output CSV
with open("inference_summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    # Loop over all models (folders in /res)
    for model_name in sorted(os.listdir(RES_ROOT)):
        model_dir = os.path.join(RES_ROOT, model_name)
        if not os.path.isdir(model_dir):
            continue

        row = [model_name]

        # Loop over all datasets
        for dataset in dataset_names:
            log_file = os.path.join(model_dir, f"{dataset}.log")
            if not os.path.exists(log_file):
                # If log missing, write empty
                row.append("0/0")
                continue

            total = 0
            no_detection = 0

            with open(log_file, "r") as f:
                for line in f:
                    if "probabilities" in line:
                        total += 1
                        if "no detection" in line.lower():
                            no_detection += 1

            detections = total - no_detection
            row.append(f"{detections}/{total}")

        writer.writerow(row)

print("CSV summary saved as inference_summary.csv")
