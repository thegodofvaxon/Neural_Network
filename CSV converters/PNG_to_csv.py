import os
import csv
import random
import numpy as np
from PIL import Image

# Path to extracted dataset folder
BASE_PATH = r"C:\Users\danie\Downloads\Projects\Data\Neural Network\Datasets\Handwritten Digits Dataset (not in MNIST)"

# Output CSV files (existing files)
TRAIN_CSV = "train1.csv"
TEST_CSV = "test1.csv"

# Collect (image_path, label) for all images
data = []
for outer_folder in sorted(os.listdir(BASE_PATH)):
    outer_path = os.path.join(BASE_PATH, outer_folder)
    if not os.path.isdir(outer_path):
        continue

    # Inner folder with same name
    inner_path = os.path.join(outer_path, outer_folder)
    if not os.path.isdir(inner_path):
        inner_path = outer_path  # fallback

    label = outer_folder.strip()
    print(f"Processing label '{label}'...")

    count = 0
    for img_name in os.listdir(inner_path):
        if img_name.lower().endswith(".png"):
            data.append((os.path.join(inner_path, img_name), label))
            count += 1
    print(f"  Found {count} images for label '{label}'")

# Shuffle and split into train/test
random.shuffle(data)
split_index = int(len(data) * 0.9)
train_data = data[:split_index]
test_data = data[split_index:]

def save_to_csv(file_path, dataset):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for img_path, label in dataset:
            img = Image.open(img_path).convert("L")  # grayscale
            pixels = np.array(img).flatten()          # keep original size
            # Create a header for the first row dynamically
            if f.tell() == 0:
                header = [f"pixel{i}" for i in range(len(pixels))] + ["label"]
                writer.writerow(header)
            writer.writerow(pixels.tolist() + [label])
    print(f"Saved {len(dataset)} rows to {file_path}")

# Save both CSVs
save_to_csv(TRAIN_CSV, train_data)
save_to_csv(TEST_CSV, test_data)
