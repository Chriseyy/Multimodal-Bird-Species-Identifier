import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("XENO_CANTO_API_KEY")

TARGET_SPECIES = [
    "Parus major",          # Kohlmeise
    "Cyanistes caeruleus",  # Blaumeise
    "Turdus merula",        # Amsel
    "Erithacus rubecula",   # Rotkehlchen
    "Fringilla coelebs",    # Buchfink
    "Passer domesticus",    # Haussperling
    "Sitta europaea",       # Kleiber
    "Pica pica",            # Elster
    "Sturnus vulgaris",     # Star
    "Dendrocopos major"     # Buntspecht

]



def download_bird_audio(species_name: str, max_files: int = 25, output_dir: str = "data/raw/audio"):
    if not API_KEY:
        print("ERROR: No XENO_CANTO_API_KEY found in .env file!")
        return

    query_str = f'sp:"{species_name}" q:A'
    
    params = {
        "query": query_str,
        "key": API_KEY,
        "per_page": min(max_files, 100)
    }
    
    url = "https://xeno-canto.org/api/3/recordings"
    headers = {"User-Agent": "Mozilla/5.0 (Student Project)"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching data for {species_name}: HTTP {response.status_code}")
            print(response.json())
            return
        
        data = response.json()
    except Exception as e:
        print(f"Request failed for {species_name}: {e}")
        return

    recordings = data.get("recordings", [])[:max_files]
    
    if not recordings:
        print(f"No recordings found for: {query_str}")
        return

    folder_name = species_name.replace(" ", "_")
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n--- Downloading {len(recordings)} tracks for: {species_name} ---")
    
    for i, rec in enumerate(recordings, start=1):
        file_url = rec.get("file")
        if not file_url:
            continue
            
        file_id = rec['id']
        file_name = f"{file_id}.mp3"
        full_file_path = os.path.join(save_path, file_name)
        
        if not os.path.exists(full_file_path):
            print(f"[{i}/{len(recordings)}] Downloading: {file_name}...")
            try:
                audio_data = requests.get(file_url, headers=headers, timeout=15).content
                with open(full_file_path, "wb") as f:
                    f.write(audio_data)
            except Exception as e:
                print(f"Failed to download {file_id}: {e}")
        else:
            print(f"[{i}/{len(recordings)}] {file_name} already exists. Skipping.")

if __name__ == "__main__":
    for species in TARGET_SPECIES:
        download_bird_audio(species, max_files=25)