import os
import glob
import torch
import torchaudio
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoFeatureExtractor

from multimodal_bird_species_identifier.models.audio_model import BirdAudioCNN
from multimodal_bird_species_identifier.models.vision_model import BirdVisionResNet
from multimodal_bird_species_identifier.models.multimodal_net import MultimodalBirdIdentifier

# 1. Species List
CLASSES = [
    "Cyanistes_caeruleus",  
    "Dendrocopos_major",    
    "Erithacus_rubecula",   
    "Fringilla_coelebs",    
    "Parus_major",          
    "Passer_domesticus",    
    "Pica_pica",            
    "Sitta_europaea",       
    "Sturnus_vulgaris",     
    "Turdus_merula"         
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load All Three Models Separately
# A. Vision Only Model
vision_model = BirdVisionResNet(num_classes=len(CLASSES))
if os.path.exists("data/vision_model.pt"):
    vision_model.load_state_dict(torch.load("data/vision_model.pt", map_location=DEVICE))
vision_model.to(DEVICE).eval()

# B. Audio Only Model
audio_model = BirdAudioCNN(num_classes=len(CLASSES))
if os.path.exists("data/audio_model.pt"):
    audio_model.load_state_dict(torch.load("data/audio_model.pt", map_location=DEVICE))
audio_model.to(DEVICE).eval()

# C. Multimodal Fusion Model
fusion_model = MultimodalBirdIdentifier(num_classes=len(CLASSES))
if os.path.exists("data/multimodal_model.pt"):
    fusion_model.load_state_dict(torch.load("data/multimodal_model.pt", map_location=DEVICE))
fusion_model.to(DEVICE).eval()

# AST Feature Extractor
ast_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# 3. Preprocessing Functions
def process_audio(audio_path):
    if audio_path is None:
        return None
        
    try:
        target_sr = 16000
        target_samples = 163840  # 10.24 seconds
        
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        speech = waveform.squeeze(0)

        if speech.shape[0] > target_samples:
            speech = speech[:target_samples]
        else:
            speech = torch.nn.functional.pad(speech, (0, target_samples - speech.shape[0]))

        inputs = ast_extractor(
            speech.numpy(), 
            sampling_rate=target_sr,
            return_tensors="pt"
        )
        return inputs.input_values.to(DEVICE)
    except Exception:
        return None

def process_image(img_pil):
    if img_pil is None:
        return None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

# 4. Prediction Pipeline
def predict_breakdown(image, audio):
    if image is None and audio is None:
        return "Please upload an image or audio file!", {}, {}

    img_tensor = process_image(image)
    audio_tensor = process_audio(audio)

    with torch.no_grad():
        vision_probs = None
        audio_probs = None
        combined_probs = None

        # 1. Independent Image Prediction
        if img_tensor is not None:
            vision_logits = vision_model(img_tensor)
            vision_probs = torch.softmax(vision_logits, dim=1)[0]
            
        # 2. Independent Audio Prediction
        if audio_tensor is not None:
            audio_logits = audio_model(audio_tensor)
            audio_probs = torch.softmax(audio_logits, dim=1)[0]

        # 3. Deep Fusion Prediction (Only if BOTH are provided)
        if img_tensor is not None and audio_tensor is not None:
            fusion_logits = fusion_model(audio_tensor, img_tensor)
            combined_probs = torch.softmax(fusion_logits, dim=1)[0]
        elif img_tensor is not None:
            # Fallback to vision if only image uploaded
            combined_probs = vision_probs
        elif audio_tensor is not None:
            # Fallback to audio if only audio uploaded
            combined_probs = audio_probs

    # Format for Gradio
    fusion_dict = {CLASSES[i].replace("_", " "): float(combined_probs[i]) for i in range(len(CLASSES))}
    
    if vision_probs is not None:
        vision_dict = {CLASSES[i].replace("_", " "): float(vision_probs[i]) for i in range(len(CLASSES))}
    else:
        vision_dict = {"No Image": 1.0}
        
    if audio_probs is not None:
        audio_dict = {CLASSES[i].replace("_", " "): float(audio_probs[i]) for i in range(len(CLASSES))}
    else:
        audio_dict = {"No Audio": 1.0}

    return fusion_dict, vision_dict, audio_dict

# 5. Collect Examples
unseen_pairs = []
seen_pairs = []

if os.path.exists("data/test_samples"):
    for img_p in glob.glob(os.path.join("data", "test_samples", "*", "*.jpg")):
        audio_p = img_p.replace(".jpg", ".mp3")
        if os.path.exists(audio_p):
            unseen_pairs.append([img_p, audio_p])

if os.path.exists("data/test_samples"):
    for img_p in glob.glob(os.path.join("data", "test_samples", "*.jpg")):
        audio_p = img_p.replace(".jpg", ".mp3")
        if os.path.exists(audio_p):
            seen_pairs.append([img_p, audio_p])

# 6. Gradio UI Layout
with gr.Blocks(title="Multimodal Bird Identifier") as app:
    gr.Markdown("# Multimodal Bird Species Identifier (ResNet + AST)")
    gr.Markdown("Upload a photo, an audio file, or both. The AI uses Independent Models for single inputs and Deep Feature Fusion when both are provided.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="1. Upload Bird Photo")
            audio_input = gr.Audio(type="filepath", label="2. Upload Bird Song / Call")
            btn = gr.Button("Identify Bird", variant="primary")

        with gr.Column():
            gr.Markdown("### Prediction Results")
            fusion_output = gr.Label(num_top_classes=3, label="Multimodal Deep Fusion (Combined)")
            
            with gr.Row():
                vision_output = gr.Label(num_top_classes=2, label="Image Features Only")
                audio_output = gr.Label(num_top_classes=2, label="Audio Features Only")

    btn.click(
        fn=predict_breakdown,
        inputs=[img_input, audio_input],
        outputs=[fusion_output, vision_output, audio_output]
    )

    gr.Markdown("---")
    gr.Markdown("### Try it yourself")
    
    with gr.Accordion("Never Seen Data (Unseen Test Samples)", open=False):
        if unseen_pairs:
            gr.Examples(
                examples=unseen_pairs,
                inputs=[img_input, audio_input],
                outputs=[fusion_output, vision_output, audio_output],
                fn=predict_breakdown,
                cache_examples=False
            )
        else:
            gr.Markdown("No unseen test samples found.")

    with gr.Accordion("Seen Data (Training Samples)", open=False):
        if seen_pairs:
            gr.Examples(
                examples=seen_pairs,
                inputs=[img_input, audio_input],
                outputs=[fusion_output, vision_output, audio_output],
                fn=predict_breakdown,
                cache_examples=False
            )
        else:
            gr.Markdown("No seen samples found.")

if __name__ == "__main__":
    app.launch()