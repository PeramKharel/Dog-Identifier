import os
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split images from source_dir (with breed subfolders) into train/val/test.
    
    Args:
        source_dir: Path to the folder containing breed subfolders (e.g., 'downloaded_breeds')
        target_dir: Path where 'train', 'validation', 'test' folders will be created
        train_ratio, val_ratio, test_ratio: Splitting ratios (must sum to 1)
    """
    # Create target directories
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(target_dir, split)
        os.makedirs(split_path, exist_ok=True)
    
    # Get all breed folders
    breeds = [d for d in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(breeds)} breeds in {source_dir}")
    
    for breed in breeds:
        breed_path = os.path.join(source_dir, breed)
        images = [f for f in os.listdir(breed_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"  ⚠️ No images in {breed}, skipping")
            continue
        
        # Shuffle images randomly
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remaining go to test (handles rounding)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create breed subfolder in each target split
        train_breed_dir = os.path.join(target_dir, 'train', breed)
        val_breed_dir = os.path.join(target_dir, 'validation', breed)
        test_breed_dir = os.path.join(target_dir, 'test', breed)
        
        os.makedirs(train_breed_dir, exist_ok=True)
        os.makedirs(val_breed_dir, exist_ok=True)
        os.makedirs(test_breed_dir, exist_ok=True)
        
        # Copy (or move) images to respective folders
        for img in train_images:
            shutil.copy2(os.path.join(breed_path, img), 
                         os.path.join(train_breed_dir, img))
        for img in val_images:
            shutil.copy2(os.path.join(breed_path, img), 
                         os.path.join(val_breed_dir, img))
        for img in test_images:
            shutil.copy2(os.path.join(breed_path, img), 
                         os.path.join(test_breed_dir, img))
        
        print(f"  ✅ {breed}: {n_total} images → "
              f"train: {len(train_images)}, val: {len(val_images)}, test: {len(test_images)}")
    
    print(f"\n🎉 Dataset split complete! Check '{target_dir}'")

if __name__ == "__main__":
    # ---- CONFIGURATION ----
    source_directory = "downloaded_breeds"   # folder with your breed subfolders
    target_directory = "data"                # where to create train/val/test
    
    # Optional: set random seed for reproducibility
    random.seed(42)
    
    # Run the split
    split_dataset(source_directory, target_directory)