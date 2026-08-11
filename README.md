# Multimodal Bird Species Identifier (ResNet-18 + AST)

A end-to-end multimodal deep learning pipeline engineered to classify 10 native European bird species using photographs, audio recordings, or both modalities simultaneously. The system integrates a vision-tailored **ResNet-18** model and an **Audio Spectrogram Transformer (AST)** through a late-stage **Deep Feature Fusion Layer**.

---

## Technical Overview & Multimodal Fusion Case Study

The core architectural objective is **robustness through modality complementarity**: when one input channel suffers from ambient noise or visual clutter, the fusion network dynamically cross-references features to resolve classification ambiguities.

### Case Study: Multimodal Error Correction in Noisy Conditions

* **Audio Branch Alone:** Predicts *Fringilla coelebs* (Common Chaffinch) with 62% confidence due to acoustic background overlap (*Sitta europaea* ranks second at 33%).
* **Vision Branch Alone:** Predicts *Sitta europaea* (Eurasian Nuthatch) with 74% confidence based on bark-crawling positioning and plumage pattern.
* **Multimodal Fusion Layer:** The joint head evaluates feature dependencies, suppresses the conflicting audio hypothesis for *Fringilla coelebs* (0%), and boosts final prediction confidence for *Sitta europaea* to **93%**.

---

## System Architecture & Pipeline Specifications

```text
[ Input: Audio (.mp3) ] ---> [ 16kHz Resample / 10.24s Pad ] ---> [ AST Transformer (AudioSet) ] ---\
                                                                                                      +---> [ Deep Fusion Classifier ] ---> Prediction (10 Classes)
[ Input: Image (.jpg) ] ---> [ RandomResizedCrop / Norm ] ------> [ ResNet-18 (Fine-Tuned) ] -------/

```

### 1. Vision Processing Branch (`vision_model.py`, `image_dataset.py`)

* **Backbone:** ResNet-18 initialized with ImageNet pre-trained weights (`ResNet18_Weights.DEFAULT`).
* **Feature Extraction:** Full unfreezing of backbone layers (`freeze_backbone=False`) to adapt convolutional filters to fine-grained avian feature boundaries.
* **Custom Classifier Head:** Replaces `resnet.fc` with a dense projection sequence: `Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10)`.
* **Augmentation Pipeline:** Training employs `RandomResizedCrop(224, scale=(0.4, 1.0))` to simulate variable subject distance, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`, and `ColorJitter(brightness=0.2, contrast=0.2)`.
* **Validation Pipeline:** Preserves aspect ratio using `Resize(256)` followed by `CenterCrop(224)` and ImageNet normalization vectors.

### 2. Audio Processing Branch (`audio_model.py`, `dataset.py`)

* **Backbone:** Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) leveraging Scaled Dot-Product Attention (`attn_implementation="sdpa"`).
* **Audio Normalization:** `BirdAudioDataset` loads MP3 files via `torchaudio`, converts multi-channel audio to mono, resamples to $16,000\text{ Hz}$, and enforces a strict duration of $10.24\text{ seconds}$ ($163,840$ samples) using zero-padding or central truncation.
* **Feature Representation:** Uses `AutoFeatureExtractor` to convert raw waveforms into guaranteed tensor dimensions of `(1024, 128)` log-mel spectrogram patches.

### 3. Deep Feature Fusion Model (`multimodal_net.py`)

* **Submodel Freezing:** Freezes parameters of both pre-trained vision and audio submodels (`param.requires_grad = False`).
* **Embedding Extraction:** Removes top classification layers (`nn.Identity()`), extracting raw 768-dimensional AST embeddings and 512-dimensional ResNet-18 embeddings.
* **Fusion Architecture ("The Referee"):** Concatenates latents into a 1280-dimensional joint feature vector and passes it through:
`Linear(1280, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.4) -> Linear(256, 10)`.
* **Training Logic (`train_fusion.py`):** `PairedMultimodalDataset` pairs each class image with a randomly sampled intra-species audio track, training *only* the fusion classifier parameters for rapid convergence.

---

## Dataset Harvesting & Zero-Leakage Isolation Strategy

Training and evaluation data are harvested programmatically via REST APIs with automated quality verification and strict dataset separation protocols.

### Data Acquisition Specifications

| Modality | Source API | Harvester Script | Processing & Integrity Verification |
| --- | --- | --- | --- |
| **Images** | iNaturalist API | `download_images.py` | Queries species Taxon IDs; filters for `quality_grade=research`; enforces minimum $500\times 500\text{px}$ resolution; converts to RGB JPEG (quality=95). |
| **Audio** | Xeno-Canto API v3 | `download_audio.py` | Queries quality grade `q:A`; filters out files $<20\text{ KB}$; validates header structure using `torchaudio.load()` to purge truncated downloads. |

### Data Leakage Prevention Strategy

To prevent optimistic evaluation metrics caused by spatial or recording overlaps:

1. **Image Test Isolation (`download_test_data.py`):** Test images query reverse chronological ordering (`order=asc` / oldest observations), whereas training scripts query standard order (`order=desc` / recent observations).
2. **Audio Test Isolation (`download_test_data.py`):** Test audio queries Page 2 (`page=2`) of Xeno-Canto results and filters IDs against an in-memory hash set of existing training files in `data/raw/audio/`.

---

## Dataset Analysis: Curated Web Search vs. Real-World Data

| Evaluation Dimension | Curated Web Search (e.g., DuckDuckGo) | Citizen-Science API (iNaturalist / Xeno-Canto) |
| --- | --- | --- |
| **Subject Positioning** | Centered, studio-like, clear visibility | Variable distance, obscured by foliage, motion blur |
| **Background Noise** | Uniform / Clean | Complex vegetation, shadow patterns, ambient audio overlap |
| **Model Behavior** | High risk of background memorization | Forces reliance on anatomical / acoustic markers |
| **Validation Benchmark** | Overfitted $>90\%$ Accuracy | **$\sim 68\%$ Single-Modality Benchmark** (Realistic "In-the-Wild") |

---

## Model Training Hyperparameters & Execution Logs

| Model | Optimizer | Base Learning Rate | Weight Decay | Epochs | Output File |
| --- | --- | --- | --- | --- | --- |
| **Vision (ResNet-18)** | AdamW | $1 \times 10^{-4}$ | $1 \times 10^{-2}$ | 20 | `data/vision_model.pt` |
| **Audio (AST Transformer)** | AdamW | $2 \times 10^{-5}$ | $1 \times 10^{-2}$ | 15 | `data/audio_model.pt` |
| **Multimodal Fusion** | AdamW | $1 \times 10^{-3}$ | $0.0$ | 5 | `data/multimodal_model.pt` |

---

## Repository Structure

```text
├── data/
│   ├── raw/images/                     # Training image directories by species
│   ├── raw/audio/                      # Training audio directories (.mp3) by species
│   └── test_samples/                   # Isolated unseen test image and audio pairs
├── src/
│   └── multimodal_bird_species_identifier/
│       ├── data/
│       │   ├── download_images.py      # iNaturalist image harvester
│       │   ├── download_audio.py       # Xeno-Canto audio harvester
│       │   ├── download_test_data.py   # Zero-leakage test dataset harvester
│       │   ├── image_dataset.py        # PyTorch ImageFolder dataset & transforms
│       │   └── dataset.py              # AST Audio Dataset & 10.24s preprocessing
│       └── models/
│           ├── vision_model.py         # ResNet-18 architecture definition
│           ├── audio_model.py          # AST Transformer model wrapper
│           ├── multimodal_net.py       # Dual-branch feature concatenation fusion net
│           └── prepare_test_samples.py # Utility script to generate Gradio sample pairs
├── train_vision.py                     # Vision branch training pipeline
├── train_audio.py                      # Audio branch fine-tuning pipeline
├── train_fusion.py                     # Fusion classifier training pipeline
├── app.py                              # Interactive Gradio web interface
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation

```

---

## Installation & Execution Guide

### 1. Environment Setup

```bash
git clone https://github.com/Chriseyy/Multimodal-Bird-Species-Identifier.git
cd Multimodal-Bird-Species-Identifier

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Dataset Acquisition

```bash
# Download training datasets
python src/multimodal_bird_species_identifier/data/download_images.py
python src/multimodal_bird_species_identifier/data/download_audio.py

# Download isolated test samples
python src/multimodal_bird_species_identifier/data/download_test_data.py

```

### 3. Sequential Model Training

```bash
python train_vision.py
python train_audio.py
python train_fusion.py

```

### 4. Interactive Web Application

```bash
python app.py

```

Open `[http://127.0.0.1:7860](http://127.0.0.1:7860)` in a web browser to evaluate the model using single or dual modal inputs.

---

## Roadmap & Potential System Improvements

### Data & Augmentation Enhancements

* **Expanded Sample Volume:** Scale training data from 100 to 300+ samples per species across both modalities to improve single-modality generalization.
* **Audio SpecAugment Pipeline:** Implement dynamic frequency masking, time masking, and background noise injection (e.g., wind/rain synthesis) into `BirdAudioDataset` to harden the AST model against environmental noise.
* **Class Expansion:** Expand target taxonomy from 10 to 50 European avian species.

### Architectural Improvements

* **YOLOv8 Bounding Box Preprocessing:** Introduce an object detection step prior to visual classification to automatically crop the bird Region of Interest (ROI), eliminating background noise from branches and foliage.
* **Cross-Attention Fusion Module:** Replace standard vector concatenation in `MultimodalBirdIdentifier` with a Cross-Attention Fusion layer, allowing visual tokens to attend directly to relevant audio spectrogram regions.
* **Model Quantization & Cloud Deployment:** Convert models to ONNX runtime format with FP16 quantization for low-latency inference and deploy the Gradio interface to Hugging Face Spaces.