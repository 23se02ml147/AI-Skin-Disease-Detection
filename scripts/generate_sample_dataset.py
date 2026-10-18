from PIL import Image, ImageDraw, ImageFont
import os
import zipfile

OUT_DIR = 'dataset_sample'
CLASSES = ['acen', 'eczema', 'psoriasis']
IMAGES_PER_CLASS = 10
IMG_SIZE = (224, 224)

os.makedirs(OUT_DIR, exist_ok=True)

for cls in CLASSES:
    cls_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for i in range(IMAGES_PER_CLASS):
        img = Image.new('RGB', IMG_SIZE, (int(255 * (CLASSES.index(cls) / len(CLASSES))),
                                          int(255 * ((i % IMAGES_PER_CLASS) / IMAGES_PER_CLASS)),
                                          100))
        d = ImageDraw.Draw(img)
        text = f"{cls} {i+1}"
        # Use default font
        try:
            fnt = ImageFont.load_default()
        except Exception:
            fnt = None
        # Compute text size using textbbox for compatibility
        try:
            bbox = d.textbbox((0, 0), text, font=fnt)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (len(text) * 6, 10)
        d.text(((IMG_SIZE[0]-text_w)/2, (IMG_SIZE[1]-text_h)/2), text, fill=(255,255,255), font=fnt)
        img_path = os.path.join(cls_dir, f"{cls}_{i+1}.jpg")
        img.save(img_path)

# Create zip
zip_name = 'dataset_sample.zip'
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            full = os.path.join(root, f)
            arc = os.path.relpath(full, start=os.path.dirname(OUT_DIR))
            zf.write(full, arc)

print('Created', zip_name)
print('Sample dataset folders:', OUT_DIR)
