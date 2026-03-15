import os
import time
import json
import requests
from icrawler.builtin import BingImageCrawler

def get_breed_classes():
    url = "https://dog.ceo/api/breeds/list/all"
    try:
        response = requests.get(url)
        data = response.json()["message"]
    except Exception as e:
        print(f"Error fetching API, using local fallback: {e}")
        with open("breeds.json") as f:
            data = json.load(f)["message"]
    
    classes = []
    for breed, sub_breeds in data.items():
        if not sub_breeds:
            classes.append(breed)
        else:
            for sub in sub_breeds:
                classes.append(f"{sub}_{breed}")
    return classes

def folder_to_query(folder_name):
    # For better search results
    return folder_name.replace('_', ' ').title()

def download_breed_images(breed_folder, download_dir, max_num=30):
    save_dir = os.path.join(download_dir, breed_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    query = folder_to_query(breed_folder)
    
    crawler = BingImageCrawler(
        storage={'root_dir': save_dir},
        downloader_threads=4,
        parser_threads=1
    )
    
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
        file_idx_offset=0
    )
    
    # Small delay
    time.sleep(2)

def main():
    start_time = time.time()

    breeds = get_breed_classes()
    print(f"Total breeds to download: {len(breeds)}")

    download_base = "downloaded_breeds"
    os.makedirs(download_base, exist_ok=True)
    
    # Testing with first 5 breeds
    # breeds = breeds[:5]   
 
    for i, breed in enumerate(breeds, 1):
        print("Processing: ", breed)
        # print(f"\n[{i}/{len(breeds)}] Processing: {breed}")
        download_breed_images(breed, download_base, max_num=10)
    
    elapsed = time.time() - start_time
    print(f"\n✅ All downloads completed in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()