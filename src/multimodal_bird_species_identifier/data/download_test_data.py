import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("XENO_CANTO_API_KEY")

TEST_DIR = "data/test_samples"

CLASSES = [
    "Cyanistes_caeruleus",  # Blue Tit
    "Dendrocopos_major",    # Great Spotted Woodpecker
    "Erithacus_rubecula",   # European Robin
    "Fringilla_coelebs",    # Common Chaffinch
    "Parus_major",          # Great Tit
    "Passer_domesticus",    # House Sparrow
    "Pica_pica",            # Eurasian Magpie
    "Sitta_europaea",       # Eurasian Nuthatch
    "Sturnus_vulgaris",     # Common Starling
    "Turdus_merula"         # Eurasian Blackbird
]

def download_test_audio(species, save_dir, count=2):
    """Downloads fresh audio files from Xeno-Canto by skipping the first 50 training samples."""
    search_name = species.replace("_", " ")
    url = "https://xeno-canto.org/api/3/recordings"
    
    headers = {"User-Agent": "Mozilla/5.0 (Student Project)"}
    
    params = {
        "query": f'sp:"{search_name}" q:A'
    }
    
    if API_KEY:
        params["key"] = API_KEY
    else:
        print("  Warning: No XENO_CANTO_API_KEY found in .env file.")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            print(f"  Error fetching API for {species}: HTTP {response.status_code}")
            return
            
        data = response.json()
        recordings = data.get("recordings", [])
        
        # Skip the first 50 results because they are used in the training set
        unseen_recordings = recordings[100:]
        
        saved = 0
        for rec in unseen_recordings:
            if saved >= count: 
                break
            
            file_url = rec.get("file")
            if not file_url: 
                continue
                
            audio_data = requests.get(file_url, headers=headers, timeout=15).content
            
            # Check if file is valid (greater than 20 KB)
            if len(audio_data) > 20000:
                file_path = os.path.join(save_dir, f"{species}_test_{saved+1}.mp3")
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                saved += 1
                print(f"  Audio {saved}/{count} saved.")
    except Exception as e:
        print(f"  Error fetching audio for {species}: {e}")


def download_test_images(species, save_dir, count=2):
    """Downloads fresh images from iNaturalist (page 3 = unseen)."""
    search_name = species.replace("_", " ")
    url = "https://api.inaturalist.org/v1/observations"
    
    params = {
        "taxon_name": search_name,
        "has[]": "photos",
        "quality_grade": "research",
        "per_page": 50,
        "page": 3 
    }
    
    try:
        response = requests.get(url, params=params, timeout=10).json()
        saved = 0
        
        for obs in response.get("results", []):
            if saved >= count: 
                break
            
            photos = obs.get("photos", [])
            if not photos: 
                continue
                
            img_url = photos[0]["url"].replace("square", "large")
            img_data = requests.get(img_url, timeout=15).content
            
            if len(img_data) > 10000:
                file_path = os.path.join(save_dir, f"{species}_test_{saved+1}.jpg")
                with open(file_path, "wb") as f:
                    f.write(img_data)
                saved += 1
                print(f"  Image {saved}/{count} saved.")
    except Exception as e:
        print(f"  Error fetching image for {species}: {e}")


def main():
    print(f"Starting download of unseen test data to '{TEST_DIR}'...\n")
    
    for cls in CLASSES:
        print(f"--- Fetching test data for: {cls.replace('_', ' ')} ---")
        
        species_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(species_dir, exist_ok=True)
        
        download_test_images(cls, species_dir, count=2)
        download_test_audio(cls, species_dir, count=2)
        
    print("\nPerfect! All real, unseen test data is ready.")

if __name__ == "__main__":
    main()