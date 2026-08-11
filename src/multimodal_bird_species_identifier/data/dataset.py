import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

class BirdAudioDataset(Dataset):
    """Dataset using ASTFeatureExtractor with strict 10.24s audio padding/truncation."""
    def __init__(self, audio_dir="data/raw/audio"):
        self.audio_dir = audio_dir
        self.classes = sorted([d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            folder = os.path.join(audio_dir, cls_name)
            for f in glob.glob(os.path.join(folder, "*.mp3")):
                self.samples.append((f, self.class_to_idx[cls_name]))

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.target_sr = 16000
        # Exactly 10.24 seconds at 16kHz = 163,840 samples
        self.target_samples = 163840

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # Squeeze to 1D array
            speech = waveform.squeeze(0)

            # Exact padding or truncation
            if speech.shape[0] > self.target_samples:
                speech = speech[:self.target_samples]
            else:
                speech = torch.nn.functional.pad(speech, (0, self.target_samples - speech.shape[0]))

            inputs = self.feature_extractor(
                speech.numpy(), 
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            
            # Returns a guaranteed shape of (1024, 128)
            return inputs.input_values.squeeze(0), label
            
        except Exception:
            # Fallback for corrupted MP3 files
            return torch.zeros((1024, 128)), label