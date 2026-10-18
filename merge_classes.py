import os
import shutil

DATASET_DIR = "dataset"
BACKUP_DIR = "dataset_original"

# Merge map
MERGES = {
    # Eczema family
    "atopic_dermatitis": "eczema",
    "contact_dermatitis": "eczema",

    # Psoriasis family
    "seborrheic_keratosis": "psoriasis",

    # Fungal
    "nail_fungus": "tinea_ringworm",

    # Bacterial
    "cellulitis_impetigo": "bullous_disease",

    # Viral
    "herpes_hpv": "warts_molluscum",

    # Skin cancer
    "actinic_keratosis_bcc": "melanoma",

    # Autoimmune
    "vasculitis": "lupus",
    "systemic_disease": "lupus",

    # Pigmentation
    "alopecia": "pigmentation_disorders",

    # Other
    "scabies_lyme": "urticaria",
    "vascular_tumors": "urticaria",
    "exanthems_drug_eruptions": "urticaria",
}

# ============================================================
print("=" * 60)
print("🔄 MERGING CLASSES")
print("=" * 60)

# Backup original dataset first
if not os.path.exists(BACKUP_DIR):
    print(f"\n📦 Backing up original dataset to {BACKUP_DIR}...")
    shutil.copytree(DATASET_DIR, BACKUP_DIR)
    print(f"✅ Backup complete")
else:
    print(f"\n⚠️  Backup exists at {BACKUP_DIR}, skipping backup")

# Merge
for src, dst in MERGES.items():
    src_path = os.path.join(DATASET_DIR, src)
    dst_path = os.path.join(DATASET_DIR, dst)

    if not os.path.exists(src_path):
        print(f"⏭️  Skipping {src} (not found)")
        continue

    os.makedirs(dst_path, exist_ok=True)
    files = os.listdir(src_path)
    for f in files:
        shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))

    try:
        os.rmdir(src_path)
    except OSError:
        pass

    print(f"✅ Merged {src} → {dst} ({len(files)} files)")

# ============================================================
print("\n" + "=" * 60)
print("📊 NEW CLASS DISTRIBUTION")
print("=" * 60)

for cls in sorted(os.listdir(DATASET_DIR)):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"   {cls}: {count} images")

print("\n✅ Done! Now retrain with: py train_working.py")