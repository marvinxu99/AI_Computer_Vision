from PIL import Image
import os

# --- Configuration ---
input_dir = "D:/dDev/AI_Computer_Vision/Projects/KinSight/family_dataset/train"
target_size = (128, 128)

# --- Step 1: Recursively scan for invalid images ---
invalid_images = []

for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, filename)
            try:
                with Image.open(img_path) as img:
                    if img.size != target_size:
                        invalid_images.append((img_path, img.size))
            except Exception as e:
                print(f"Could not open {img_path}: {e}")

# --- Step 2: Report results ---
if invalid_images:
    print(f"\nFound {len(invalid_images)} images not sized {target_size}:\n")
    for path, size in invalid_images:
        print(f" - {path}: {size}")
else:
    print(f"\nAll images in {input_dir} are already {target_size}!")
    exit()

# --- Step 3: Ask for confirmation before resizing ---
confirm = input(f"\nDo you want to resize these images to {target_size}? (y/n): ").strip().lower()

if confirm == 'y':
    for path, _ in invalid_images:
        try:
            # Build mirrored "resized_fixed" output path
            relative_path = os.path.relpath(path, input_dir)
            resized_dir = os.path.join(input_dir, "resized_fixed", os.path.dirname(relative_path))
            os.makedirs(resized_dir, exist_ok=True)

            # Resize and save
            img = Image.open(path)
            img = img.resize(target_size)
            save_path = os.path.join(resized_dir, os.path.basename(path))
            img.save(save_path)
        except Exception as e:
            print(f"Could not resize {path}: {e}")

    print(f"\nDone! Resized images saved under: {os.path.join(input_dir, 'resized_fixed')}")
else:
    print("\nNo changes made.")
