import os
import time
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS

TARGET_SPECIES = [
    "Parus major",         
    "Cyanistes caeruleus",  
    "Turdus merula",       
    "Erithacus rubecula",  
    "Fringilla coelebs",   
    "Passer domesticus",    
    "Sitta europaea",       
    "Pica pica",          
    "Sturnus vulgaris",    
    "Dendrocopos major"    
]

BAD_WORDS = ["egg", "eier", "nest", "draw", "illustration", "diagram", "skull"]

def download_bird_images(species_name: str, max_images: int = 30, output_dir: str = "data/raw/images"):
    folder_name = species_name.replace(" ", "_")
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    if len(existing_files) >= max_images:
        print(f"Folder for {species_name} already contains {len(existing_files)} images. Skipping.")
        return

    print(f"\n--- Starting image search for: {species_name} ---")

    query = f"{species_name} bird photo"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_images + 20))
    except Exception as e:
        print(f"Search error for {species_name}: {e}")
        return

    if not results:
        print(f"No results found for: {species_name}")
        return

    saved_count = len(existing_files)
    for item in results:
        if saved_count >= max_images:
            break

        img_url = item.get("image", "")
        img_url_lower = img_url.lower()

        if any(bad_word in img_url_lower for bad_word in BAD_WORDS):
            continue

        try:
            resp = requests.get(img_url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue

            img = Image.open(BytesIO(resp.content))
            width, height = img.size

            if width < 300 or height < 300:
                continue

            saved_count += 1
            file_name = f"{saved_count:03d}.jpg"
            full_path = os.path.join(save_path, file_name)

            img.convert("RGB").save(full_path, "JPEG")
            print(f"[{saved_count}/{max_images}] Saved: {file_name} ({width}x{height}px)")

        except Exception:
            continue

    print(f"-> Completed for {species_name}: {saved_count} images available.")


if __name__ == "__main__":
    for species in TARGET_SPECIES:
        download_bird_images(species, max_images=50)
        # 3-second pause between species to avoid rate limits
        time.sleep(3)