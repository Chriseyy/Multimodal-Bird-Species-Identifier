import os
import time
import requests
from PIL import Image
from io import BytesIO

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

def get_taxon_id(species_name):
    """Fetches the official iNaturalist Taxon ID."""
    url = "https://api.inaturalist.org/v1/taxa"
    headers = {"User-Agent": "MultimodalBirdIdentifier/1.0 (Student Project)"}
    params = {"q": species_name, "per_page": 1}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0].get("id")
    except Exception as e:
        print(f"Error fetching Taxon ID for {species_name}: {e}")
    return None

def download_bird_images(species_name: str, max_images: int = 100, output_dir: str = "data/raw/images"):
    folder_name = species_name.replace(" ", "_")
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    saved_count = len(existing_files)
    
    if saved_count >= max_images:
        print(f"Folder for {species_name} already contains {saved_count} images. Skipping.")
        return

    print(f"\n--- Looking up Taxon ID for: {species_name} ---")
    taxon_id = get_taxon_id(species_name)
    
    if not taxon_id:
        print(f"Could not find a valid Taxon ID for {species_name}. Skipping.")
        return
        
    print(f"Found Taxon ID: {taxon_id}. Starting image download...")
    
    url = "https://api.inaturalist.org/v1/observations"
    headers = {"User-Agent": "MultimodalBirdIdentifier/1.0 (Student Project)"}
    
    per_page = 50
    page = 1
    
    while saved_count < max_images:
        params = {
            "taxon_id": taxon_id,
            "has[]": "photos",
            "quality_grade": "research",
            "per_page": per_page,
            "page": page
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                print(f"Error fetching API for {species_name}: HTTP {response.status_code}")
                break
                
            data = response.json()
            observations = data.get("results", [])
        except Exception as e:
            print(f"Search error for {species_name} on page {page}: {e}")
            break

        if not observations:
            print(f"No more observations found for {species_name} on page {page}.")
            break

        for obs in observations:
            if saved_count >= max_images:
                break

            photos = obs.get("photos", [])
            if not photos:
                continue

            img_url = photos[0].get("url", "").replace("square", "large")
            if not img_url:
                continue

            try:
                resp = requests.get(img_url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue

                img = Image.open(BytesIO(resp.content))
                width, height = img.size

                if width < 500 or height < 500:
                    continue

                saved_count += 1
                file_name = f"{saved_count:03d}.jpg"
                full_path = os.path.join(save_path, file_name)

                img.convert("RGB").save(full_path, "JPEG", quality=95)
                print(f"[{saved_count}/{max_images}] Saved: {file_name} ({width}x{height}px)")

            except Exception:
                continue

        page += 1
        time.sleep(0.5)

    print(f"-> Completed for {species_name}: {saved_count} images available.")


if __name__ == "__main__":
    os.makedirs("data/raw/images", exist_ok=True)
    
    for species in TARGET_SPECIES:
        download_bird_images(species, max_images=100)
        time.sleep(1)