import os

DATASET_DIR = "dataset"

print("=" * 60)
print("📊 DATASET VALIDATION")
print("=" * 60)

classes = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])

empty = []
populated = []

for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    count = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
    if count == 0:
        empty.append(cls)
    else:
        populated.append((cls, count))

print(f"\n✅ Populated classes: {len(populated)}")
for cls, count in populated:
    print(f"   {cls}: {count}")

if empty:
    print(f"\n❌ EMPTY classes: {len(empty)}")
    for cls in empty:
        print(f"   {cls}")

print(f"\n{'⚠️  FIX EMPTY CLASSES BEFORE TRAINING!' if empty else '✅ Dataset is OK!'}")