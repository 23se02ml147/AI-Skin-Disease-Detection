import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("🚀 FAST FEATURE EXTRACTION TRAINING (TARGET 85%+)")
print("=" * 60)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset_final" 

if not os.path.exists(DATASET_DIR):
    print(f"❌ ERROR: Folder '{DATASET_DIR}' not found. Please run the dataset preparation script first.")
    exit()

# 1. Load Data (NO rescale here)
datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
    class_mode="sparse", subset="training", shuffle=False
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
    class_mode="sparse", subset="validation", shuffle=False
)

classes = list(train_gen.class_indices.keys())
with open("classes.json", "w") as f: 
    json.dump(classes, f)
print(f"\n✅ Found {len(classes)} classes: {classes}")

# 2. Build the Feature Extractor
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3)
)
base_model.trainable = False

# 3. Extract Features
print("\n⏳ Extracting features from training images (Please wait 1-2 minutes)...")
train_features = base_model.predict(train_gen, verbose=1)
train_labels = train_gen.classes

print("\n⏳ Extracting features from validation images...")
val_features = base_model.predict(val_gen, verbose=1)
val_labels = val_gen.classes

# 4. Build a Fast Classifier Head
print("\n🧠 Training High-Accuracy Classifier...")
model = keras.Sequential([
    layers.Input(shape=(1280,)), 
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Calculate Class weights
class_weights = compute_class_weight(
    class_weight="balanced", 
    classes=np.unique(train_labels), 
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

# Train the classifier (No Phase 1 or Phase 2!)
history = model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]
)

# 5. Final Results
loss, acc = model.evaluate(val_features, val_labels, verbose=0)
print("\n" + "=" * 60)
print(f"🏆 FINAL ACCURACY: {acc * 100:.2f}%")
print("=" * 60)

# Save the classifier
model.save("skin_model_head.keras")
print("✅ Model head saved as 'skin_model_head.keras'")