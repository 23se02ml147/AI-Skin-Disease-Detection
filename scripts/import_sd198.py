#!/usr/bin/env python3
"""Import SD-198 (or similar) into the project's `dataset/` layout.

Usage:
    python scripts/import_sd198.py --src PATH/TO/SD-198 --dst dataset --limit 200

If `--limit` is given it will copy at most that many images per class (useful for balancing).
"""
import os
import shutil
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def copy_dataset(src: str, dst: str, limit: int | None = None):
    os.makedirs(dst, exist_ok=True)
    classes = []
    for entry in sorted(os.listdir(src)):
        src_path = os.path.join(src, entry)
        if not os.path.isdir(src_path):
            continue
        class_name = entry.strip()
        safe_name = class_name.lower().replace(' ', '_')
        dest_path = os.path.join(dst, safe_name)
        os.makedirs(dest_path, exist_ok=True)
        copied = 0
        # Walk in case images are nested in subfolders
        for root, _, files in os.walk(src_path):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    if limit is not None and copied >= limit:
                        break
                    src_file = os.path.join(root, f)
                    # Make destination filename unique
                    base, ext = os.path.splitext(f)
                    dst_file = os.path.join(dest_path, f)
                    if os.path.exists(dst_file):
                        dst_file = os.path.join(dest_path, f"{base}_{copied}{ext}")
                    shutil.copy2(src_file, dst_file)
                    copied += 1
            if limit is not None and copied >= limit:
                break
        classes.append((safe_name, copied))
        print(f"Copied {copied} images into {dest_path}")
    print("Import complete. Classes imported:")
    for c, cnt in classes:
        print(f" - {c}: {cnt}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Path to extracted SD-198 folder (contains class subfolders)')
    p.add_argument('--dst', default='dataset', help='Destination dataset folder (will be created)')
    p.add_argument('--limit', type=int, default=None, help='Max images to copy per class')
    args = p.parse_args()
    copy_dataset(args.src, args.dst, args.limit)
