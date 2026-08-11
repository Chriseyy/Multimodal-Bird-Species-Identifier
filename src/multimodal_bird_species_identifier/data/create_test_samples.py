import os
import shutil
import glob

SOURCE_IMAGE_DIR = "data/raw/images"
SOURCE_AUDIO_DIR = "data/raw/audio"
TEST_SAMPLES_DIR = "data/test_samples"

def prepare_test_samples(samples_per_class=2):
    """Erstellt eine Auswahl von Test-Bildern und -Audios für jede Vogelart."""
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)
    
    # Klassenliste aus den Ordnern holen
    if not os.path.exists(SOURCE_IMAGE_DIR):
        print(f"Fehler: Ordner {SOURCE_IMAGE_DIR} existiert nicht!")
        return

    classes = sorted([d for d in os.listdir(SOURCE_IMAGE_DIR) if os.path.isdir(os.path.join(SOURCE_IMAGE_DIR, d))])
    
    copied_count = 0
    test_examples_list = []

    print("--- Erstelle Test-Samples für Gradio ---")

    for cls in classes:
        img_folder = os.path.join(SOURCE_IMAGE_DIR, cls)
        audio_folder = os.path.join(SOURCE_AUDIO_DIR, cls)
        
        # Finde Bild- und Audio-Dateien
        images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
        audios = sorted(glob.glob(os.path.join(audio_folder, "*.mp3")))
        
        count = min(len(images), len(audios), samples_per_class)
        
        for i in range(count):
            img_src = images[i]
            audio_src = audios[i]
            
            img_dst = os.path.join(TEST_SAMPLES_DIR, f"{cls}_sample_{i+1}.jpg")
            audio_dst = os.path.join(TEST_SAMPLES_DIR, f"{cls}_sample_{i+1}.mp3")
            
            shutil.copy(img_src, img_dst)
            shutil.copy(audio_src, audio_dst)
            
            test_examples_list.append([img_dst, audio_dst])
            copied_count += 1
            print(f"[{cls}] Sample {i+1} kopiert -> {img_dst} & {audio_dst}")

    print(f"\nFertig! Insgesamt {copied_count} Test-Paare in '{TEST_SAMPLES_DIR}' bereitgestellt.")
    return test_examples_list

if __name__ == "__main__":
    prepare_test_samples(samples_per_class=2)