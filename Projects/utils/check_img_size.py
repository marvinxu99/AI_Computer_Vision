from PIL import Image
import os

# input_dir = "images/"  # change to your folder path
input_dir =  "D:/dDev/AI_Computer_Vision/Projects/KinSight/family_dataset/train/Eric"
target_size = (128, 128)

# --- Step 1: Scan for invalid images ---
invalid_images = []

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        try:
            with Image.open(img_path) as img:
                if img.size != target_size:
                    invalid_images.append((filename, img.size))
        except Exception as e:
            print(f"Could not open {filename}: {e}")

# --- Step 2: Report results ---
if invalid_images:
    print(f"\n Found {len(invalid_images)} images not sized { target_size}:\n")
    for name, size in invalid_images:
        print(f" - {name}: {size}")
else:
    print(f"\n All images are already { target_size}!")
    exit()

# --- Step 3: Ask for confirmation before resizing ---
confirm = input(f"\nDo you want to resize these images to { target_size }? (y/n): ").strip().lower()

if confirm == 'y':
    resized_dir = os.path.join(input_dir, "resized_fixed")
    os.makedirs(resized_dir, exist_ok=True)

    for name, _ in invalid_images:
        img_path = os.path.join(input_dir, name)
        try:
            img = Image.open(img_path)
            img = img.resize(target_size)
            img.save(os.path.join(resized_dir, name))
        except Exception as e:
            print(f"Could not resize {name}: {e}")

    print(f"\nDone! Resized images saved in: {resized_dir}")
else:
    print("\nNo changes made.")