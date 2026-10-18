import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import defaultdict

def load_model_and_labels():
    candidates = ["best_skin_model.keras", "skin_model_transfer.keras", "skin_model.keras", "skin_model.h5"]
    model = None
    for c in candidates:
        if os.path.exists(c):
            try:
                model = tf.keras.models.load_model(c)
                print("Loaded model:", c)
                break
            except Exception as e:
                print("Failed to load", c, e)
    if model is None:
        raise RuntimeError("No model file found. Run train.py first.")
    if os.path.exists('classes.json'):
        with open('classes.json','r',encoding='utf-8') as f:
            classes = json.load(f)
    else:
        classes = ["acen","eczema","psoriasis"]
    return model, classes

def evaluate(dataset_dir='dataset'):
    model, classes = load_model_and_labels()
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_true = []
    y_pred = []
    probs_all = []
    examples_wrong = []

    for cls in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fn in os.listdir(cls_path):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            path = os.path.join(cls_path, fn)
            try:
                img = Image.open(path).convert('RGB').resize((224,224))
            except Exception as e:
                print('Failed to open', path, e)
                continue
            arr = np.array(img, dtype=np.float32)/255.0
            arr = np.expand_dims(arr,0)
            pred = model.predict(arr, verbose=0)[0]
            pred_idx = int(np.argmax(pred))
            probs_all.append(pred)
            y_true.append(cls)
            pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            y_pred.append(pred_label)
            if pred_label != cls:
                examples_wrong.append((path, cls, pred_label, pred))

    # Summaries
    print('\nTotal samples:', len(y_true))
    if len(y_true) == 0:
        print('No images found in', dataset_dir)
        return

    # Per-class counts
    counts = defaultdict(int)
    correct = defaultdict(int)
    for t,p in zip(y_true,y_pred):
        counts[t]+=1
        if t==p:
            correct[t]+=1

    print('\nPer-class results:')
    for cls in sorted(counts.keys()):
        print(f" - {cls}: {correct[cls]}/{counts[cls]} correct ({(correct[cls]/counts[cls]*100):.1f}%)")

    # Confusion-like summary
    labels = sorted(list(set(list(counts.keys())+list(classes))))
    idx_map = {l:i for i,l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true,y_pred):
        cm[idx_map[t], idx_map[p]] += 1

    print('\nConfusion matrix rows=true cols=pred:')
    print('labels:', labels)
    print(cm)

    # Show some wrong examples
    print('\nExamples of wrong predictions (up to 10):')
    for ex in examples_wrong[:10]:
        path, true, pred_label, pred = ex
        top3 = sorted(list(enumerate(pred)), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ', '.join([f"{classes[i]}:{v*100:.1f}%" for i,v in top3])
        print(f"{path} -> true={true}, pred={pred_label}, top3=[{top3_str}]")

if __name__ == '__main__':
    evaluate()
