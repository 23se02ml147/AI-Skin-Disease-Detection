import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

print("=" * 70)
print("🔬 SKIN DISEASE DETECTION — EfficientNetB0 Training")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16                # smaller = better generalization
EPOCHS_PHASE_1 = 15
EPOCHS_PHASE_2 = 20
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "best_skin_model.keras"
FINAL_MODEL_PATH = "skin_model.keras"
CLASSES_FILE = "classes.json"
SEED = 42

# ============================================================
# CHECK DATASET
# ============================================================
print("\n📂 Checking dataset...")

if not os.path.exists(DATASET_DIR):
    print(f"❌ Dataset folder '{DATASET_DIR}' not found!")
    exit(1)

classes = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"✅ Found {len(classes)} classes")

total_images = 0
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    count = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
    total_images += count
    print(f"   {cls}: {count}")

print(f"\n✅ Total images: {total_images}")

# ============================================================
# DATA GENERATORS
# ============================================================
print("\n🔄 Setting up data generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # EfficientNet preprocessing
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print(f"\n✅ Train: {train_generator.samples}, Val: {validation_generator.samples}")
print(f"✅ Classes: {len(train_generator.class_indices)}")

# ============================================================
# SAVE CLASSES (CORRECT ORDER)
# ============================================================
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

with open(CLASSES_FILE, 'w') as f:
    json.dump(class_names, f)
print(f"✅ Class names saved (ordered): {class_names[:3]}...")

# ============================================================
# BUILD MODEL — EfficientNetB0
# ============================================================
print("\n🧠 Building EfficientNetB0 model...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),   # LOWER LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ============================================================
# CLASS WEIGHTS (mild)
# ============================================================
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: min(float(w), 2.0) for i, w in enumerate(weights)}
print(f"\n📊 Class weights capped at 2.0: {class_weight_dict}")

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, mode='max', verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
]

# ============================================================
# PHASE 1 — Frozen base
# ============================================================
print("\n" + "=" * 70)
print(f"PHASE 1: Training top layers ({EPOCHS_PHASE_1} epochs)")
print("=" * 70)

model.fit(
    train_generator, validation_data=validation_generator,
    epochs=EPOCHS_PHASE_1, callbacks=callbacks,
    class_weight=class_weight_dict, verbose=1
)

# ============================================================
# PHASE 2 — Fine-tune
# ============================================================
print("\n" + "=" * 70)
print(f"PHASE 2: Fine-tuning ({EPOCHS_PHASE_2} epochs)")
print("=" * 70)

base_model.trainable = True
# Freeze first 100 layers, unfreeze the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # very low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator, validation_data=validation_generator,
    epochs=EPOCHS_PHASE_2, callbacks=callbacks,
    class_weight=class_weight_dict, verbose=1
)

# ============================================================
# SAVE
# ============================================================
model.save(FINAL_MODEL_PATH)
print(f"\n💾 Final model saved to {FINAL_MODEL_PATH}")

# ============================================================
# EVALUATE
# ============================================================
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
val_loss, val_acc = best_model.evaluate(validation_generator, verbose=0)

print("\n" + "=" * 70)
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")
print("=" * 70)

# ============================================================
# CLASS REPORT
# ============================================================
from sklearn.metrics import classification_report
val_preds = best_model.predict(validation_generator, verbose=0)
print(classification_report(
    validation_generator.classes,
    np.argmax(val_preds, axis=1),
    target_names=class_names,
    zero_division=0
))

if val_acc < 0.4:
    print("\n⚠️  Still low. Next step: merge similar classes.")
elif val_acc < 0.6:
    print("\n✅ Moderate. Ready for demo.")
else:
    print("\n🎉 Great! Run: py -m streamlit run app.py")