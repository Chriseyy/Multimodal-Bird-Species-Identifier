import os
import requests
import torchaudio
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

def is_valid_audio(file_path: str) -> bool:
    """Robust verification to check if the downloaded MP3 can be opened by PyTorch."""
    if not os.path.exists(file_path):
        return False
        
    # Standard MP3s from Xeno-Canto are always > 20 KB
    if os.path.getsize(file_path) < 20000:
        return False
        
    try:
        # Load header using torchaudio default loader (without forcing backend="soundfile")
        waveform, sr = torchaudio.load(file_path)
        return waveform.numel() > 0
    except Exception:
        return False


def download_bird_audio(species_name: str, max_files: int = 50, output_dir: str = "data/raw/audio"):
    if not API_KEY:
        print("ERROR: No XENO_CANTO_API_KEY found in .env file!")
        return

    folder_name = species_name.replace(" ", "_")
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # 1. Check existing valid files in directory
    existing_files = [f for f in os.listdir(save_path) if f.endswith('.mp3')]
    valid_saved = 0
    for f in existing_files:
        full_p = os.path.join(save_path, f)
        if is_valid_audio(full_p):
            valid_saved += 1
        else:
            # Only delete if file is genuinely broken or HTML error
            os.remove(full_p)

    if valid_saved >= max_files:
        print(f"Folder for {species_name} already has {valid_saved}/{max_files} valid tracks. Skipping.")
        return

    print(f"\n--- Downloading tracks for: {species_name} (Current: {valid_saved}/{max_files}) ---")

    # 2. Fetch up to 100 recordings from Xeno-Canto API to ensure we find enough valid ones
    query_str = f'sp:"{species_name}" q:A'
    params = {
        "query": query_str,
        "key": API_KEY,
        "per_page": 100
    }
    
    url = "https://xeno-canto.org/api/3/recordings"
    headers = {"User-Agent": "Mozilla/5.0 (Student Project)"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching data for {species_name}: HTTP {response.status_code}")
            return
        
        data = response.json()
    except Exception as e:
        print(f"Request failed for {species_name}: {e}")
        return

    recordings = data.get("recordings", [])
    if not recordings:
        print(f"No recordings found for: {query_str}")
        return

    # 3. Download until we reach max_files valid tracks
    for rec in recordings:
        if valid_saved >= max_files:
            break

        file_url = rec.get("file")
        if not file_url:
            continue
            
        file_id = rec['id']
        file_name = f"{file_id}.mp3"
        full_file_path = os.path.join(save_path, file_name)

        if os.path.exists(full_file_path):
            continue

        try:
            audio_data = requests.get(file_url, headers=headers, timeout=15).content
            with open(full_file_path, "wb") as f:
                f.write(audio_data)

            # Check if downloaded file is genuinely usable
            if is_valid_audio(full_file_path):
                valid_saved += 1
                print(f"[{valid_saved}/{max_files}] Saved & Verified: {file_name}")
            else:
                print(f"File {file_name} failed audio verification. Deleting & fetching next...")
                if os.path.exists(full_file_path):
                    os.remove(full_file_path)

        except Exception as e:
            print(f"Failed to download {file_id}: {e}")
            if os.path.exists(full_file_path):
                os.remove(full_file_path)

    print(f"-> Completed {species_name}: {valid_saved}/{max_files} valid tracks ready.")


if __name__ == "__main__":
    for species in TARGET_SPECIES:
        download_bird_audio(species, max_files=100)