import os
import shutil

DATASET_DIR = "dataset"
BACKUP_DIR = "dataset_10class_backup"

FINAL_MERGES = {
    "eczema": "inflammatory",
    "psoriasis": "inflammatory",
    "lupus": "inflammatory",
    "urticaria": "inflammatory",
    "melanoma": "pigmentation",
    "pigmentation_disorders": "pigmentation",
    "tinea_ringworm": "fungal",
    "acne_rosacea": "acne_bacterial",
    "bullous_disease": "acne_bacterial",
    "warts_molluscum": "viral",
}

print("=" * 60)
print("🔄 FINAL MERGE: 10 classes → 5 categories")
print("=" * 60)

# Backup
if not os.path.exists(BACKUP_DIR):
    print(f"\n📦 Backing up to {BACKUP_DIR}...")
    shutil.copytree(DATASET_DIR, BACKUP_DIR)
    print("✅ Backup done")
else:
    print(f"\n⚠️  Backup exists at {BACKUP_DIR}")

# Merge
for src, dst in FINAL_MERGES.items():
    src_path = os.path.join(DATASET_DIR, src)
    dst_path = os.path.join(DATASET_DIR, dst)

    if not os.path.exists(src_path):
        print(f"⏭️  Skipping {src} (missing)")
        continue

    os.makedirs(dst_path, exist_ok=True)
    files = os.listdir(src_path)

    for f in files:
        shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))

    try:
        os.rmdir(src_path)
    except OSError:
        pass

    print(f"✅ {src} → {dst} ({len(files)} files)")

# Report
print("\n" + "=" * 60)
print("📊 FINAL DISTRIBUTION")
print("=" * 60)

for cls in sorted(os.listdir(DATASET_DIR)):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"   {cls}: {count} images")

print("\n✅ Done! Now retrain: py train_working.py")