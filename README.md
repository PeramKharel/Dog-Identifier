## Dog Identifier

This project prepares a dog-breed image dataset and (optionally) a model for identifying dog breeds.  
It uses the public Dog CEO API for breed names and Bing image search (via `icrawler`) to download images.

### Prerequisites

- **Python**: 3.9+ recommended  
- **Dependencies**:
  - `requests`
  - `icrawler`

Install dependencies (from the project root):

```bash
pip install requests icrawler
```

### 1. Download images (run this first)

This step pulls images from the web for each breed.

From the project root, run:

```bash
python download_breed_images.py
```

- **Download location**: images are saved under the `downloaded_breeds/` folder, with one subfolder per breed.
- **Number of images per breed**: controlled by the `max_num` argument in `download_breed_images.py`:
  - In `main`, you will see:
    ```python
    download_breed_images(breed, download_base, max_num=10)
    ```
  - **Feel free to change `max_num`** to any value you like (for example, `max_num=30` or `max_num=50`) to download more or fewer images per breed.
- **Limiting breeds for testing**:
  - In `main`, you can temporarily enable:
    ```python
    # breeds = breeds[:5]
    ```
  - Uncomment and adjust this slice to download images for only a subset of breeds while testing.

### 2. Create dataset folder structure

Once you have images downloaded (step 1), you can create a dataset folder structure for training/validation/testing:

```bash
python fetchBreedList.py
```

This script:

- Fetches the list of dog breeds from the Dog CEO API.
- Generates combined breed names (e.g., `boston_bulldog`).
- Creates the following directories under `data/`:
  - `data/train/<breed_name>/`
  - `data/validation/<breed_name>/`
  - `data/test/<breed_name>/`

You can then manually or programmatically move/copy images from `downloaded_breeds/` into these folders as needed for your training pipeline.

### 3. Dog identifier model (future work)

The `dog_identifier.py` file is a placeholder for your model training and inference logic.  
You can use the folders created in `data/` (with images you downloaded and organized) as input to your model.

