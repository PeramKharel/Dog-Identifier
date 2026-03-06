import os
import json
import requests

# 1. Get the breed data
url = "https://dog.ceo/api/breeds/list/all"
response = requests.get(url)
data = response.json()["message"]

# 2. Generate class names
classes = []
for breed, sub_breeds in data.items():
    if not sub_breeds:
        classes.append(breed)  # e.g., "affenpinscher"
    else:
        for sub in sub_breeds:
            # Combine: e.g., "boston" + "bulldog" -> "boston_bulldog"
            class_name = f"{sub}_{breed}"
            classes.append(class_name)

# print(f"Total classes: {len(classes)}")
# print("Sample classes:", classes[:10])

# 3. Create folder structure
BASE_DIR = "data"
splits = ["train", "validation", "test"]

for split in splits:
    split_path = os.path.join(BASE_DIR, split)
    os.makedirs(split_path, exist_ok=True)
    for class_name in classes:
        class_folder = os.path.join(split_path, class_name)
        os.makedirs(class_folder, exist_ok=True)

print("Folder structure created successfully.")