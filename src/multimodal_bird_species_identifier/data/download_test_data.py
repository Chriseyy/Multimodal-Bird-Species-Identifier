import os
import requests
import torchaudio
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("XENO_CANTO_API_KEY")

TEST_DIR = "data/test_samples"
TRAIN_AUDIO_DIR = "data/raw/audio"

CLASSES = [
    "Cyanistes_caeruleus",  # Blaumeise
    "Dendrocopos_major",    # Buntspecht
    "Erithacus_rubecula",   # Rotkehlchen
    "Fringilla_coelebs",    # Buchfink
    "Parus_major",          # Kohlmeise
    "Passer_domesticus",    # Haussperling
    "Pica_pica",            # Elster
    "Sitta_europaea",       # Kleiber
    "Sturnus_vulgaris",     # Star
    "Turdus_merula"         # Amsel
]

def is_valid_audio(file_path: str) -> bool:
    """Robust verification to check if downloaded audio can be opened by torchaudio."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 20000:
        return False
    try:
        waveform, _ = torchaudio.load(file_path)
        return waveform.numel() > 0
    except Exception:
        return False

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
        print(f"  [Taxon ID Error] {species_name}: {e}")
    return None

def download_test_audio(species, save_dir, count=2):
    """Downloads fresh, unseen test audio from Xeno-Canto using Page 2."""
    if not API_KEY:
        print("  [Audio Error] No XENO_CANTO_API_KEY found in .env!")
        return

    species_clean = species.replace("_", " ")
    url = "https://xeno-canto.org/api/3/recordings"
    headers = {"User-Agent": "Mozilla/5.0 (Student Project)"}
    
    # WICHTIG: "page": 2 holt die Aufnahmen 101–200 (keine Überschneidung mit Training!)
    params = {
        "query": f'sp:"{species_clean}" q:A',
        "key": API_KEY,
        "per_page": 100,
        "page": 2
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            print(f"  [Audio] HTTP Error {response.status_code} for {species}")
            return
            
        recordings = response.json().get("recordings", [])
        
        # Falls eine Art auf Seite 2 keine A-Aufnahmen mehr hat, versuche Seite 1 mit Qualität B
        if not recordings:
            params["query"] = f'sp:"{species_clean}" q:B'
            params["page"] = 1
            response = requests.get(url, headers=headers, params=params, timeout=10)
            recordings = response.json().get("recordings", [])

        train_species_dir = os.path.join(TRAIN_AUDIO_DIR, species)
        train_files = set(os.listdir(train_species_dir)) if os.path.exists(train_species_dir) else set()

        saved = 0

        for rec in recordings:
            if saved >= count:
                break
            
            rec_id = rec.get("id")
            if f"{rec_id}.mp3" in train_files:
                continue

            file_url = rec.get("file")
            if not file_url:
                continue
                
            if file_url.startswith("//"):
                file_url = "https:" + file_url

            file_path = os.path.join(save_dir, f"{species}_test_{saved+1}.mp3")
            
            try:
                audio_data = requests.get(file_url, headers=headers, timeout=15).content
                with open(file_path, "wb") as f:
                    f.write(audio_data)

                if is_valid_audio(file_path):
                    saved += 1
                    print(f"  [Audio] Saved test track {saved}/{count}: {species}_test_{saved}.mp3")
                else:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)

    except Exception as e:
        print(f"  [Audio] Error for {species}: {e}")

def download_test_images(species, save_dir, count=2):
    """Downloads fresh test images using Taxon ID and reverse order (asc)."""
    search_name = species.replace("_", " ")
    taxon_id = get_taxon_id(search_name)
    
    if not taxon_id:
        print(f"  [Image] Could not find Taxon ID for {species}")
        return

    url = "https://api.inaturalist.org/v1/observations"
    headers = {"User-Agent": "MultimodalBirdIdentifier/1.0 (Student Project)"}
    
    params = {
        "taxon_id": taxon_id,
        "has[]": "photos",
        "quality_grade": "research",
        "per_page": 20,
        "order": "asc",
        "order_by": "created_at"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            print(f"  [Image] HTTP Error {response.status_code} for {species}")
            return

        results = response.json().get("results", [])
        saved = 0
        
        for obs in results:
            if saved >= count: 
                break
            
            photos = obs.get("photos", [])
            if not photos: 
                continue
                
            img_url = photos[0].get("url", "").replace("square", "large")
            if not img_url:
                continue

            try:
                img_data = requests.get(img_url, headers=headers, timeout=15).content
                if len(img_data) > 10000:
                    file_path = os.path.join(save_dir, f"{species}_test_{saved+1}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(img_data)
                    saved += 1
                    print(f"  [Image] Saved test image {saved}/{count}: {species}_test_{saved}.jpg")
            except Exception:
                continue

        if saved == 0:
            print(f"  [Image] Could not save any images for {species}")

    except Exception as e:
        print(f"  [Image] Error for {species}: {e}")


def main():
    print(f"Starting download of unseen test data to '{TEST_DIR}'...\n")
    
    for cls in CLASSES:
        print(f"--- Fetching test data for: {cls.replace('_', ' ')} ---")
        species_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(species_dir, exist_ok=True)
        
        download_test_images(cls, species_dir, count=2)
        download_test_audio(cls, species_dir, count=2)
        print()
        
    print("Test data download finished!")

if __name__ == "__main__":
    main()