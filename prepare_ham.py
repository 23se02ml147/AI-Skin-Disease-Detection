import os
import shutil
import pandas as pd
import kagglehub

print("Downloading HAM10000 dataset...")
path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
print(f"Downloaded to: {path}")

# Find metadata CSV
csv_file = None
for root, dirs, files in os.walk(path):
    for f in files:
        if f == 'HAM10000_metadata.csv':
            csv_file = os.path.join(root, f)
            break
    if csv_file:
        break

if not csv_file:
    print("ERROR: metadata CSV not found!")
    exit(1)

df = pd.read_csv(csv_file)
print(f"Total images: {len(df)}")

# Class mapping
label_map = {
    'akiec': 'actinic_keratosis',
    'bcc': 'basal_cell_carcinoma',
    'bkl': 'benign_keratosis',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'nevus',
    'vasc': 'vascular_lesion'
}

output_dir = 'dataset_ham'
os.makedirs(output_dir, exist_ok=True)

# Create folders
for label in label_map.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Build image_id -> label map
id_to_label = dict(zip(df['image_id'], df['dx']))

# Copy images
count = 0
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.jpg'):
            image_id = f.replace('.jpg', '')
            if image_id in id_to_label:
                label = label_map[id_to_label[image_id]]
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, label, f)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    count += 1

print(f"\nCopied {count} images to {output_dir}/")
print("\nClass distribution:")
for cls in sorted(os.listdir(output_dir)):
    cls_path = os.path.join(output_dir, cls)
    if os.path.isdir(cls_path):
        print(f"  {cls}: {len(os.listdir(cls_path))} images")