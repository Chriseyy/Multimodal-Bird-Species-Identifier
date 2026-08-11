import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# Force soundfile as the global backend for torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass


class BirdAudioDataset(Dataset):
    def __init__(self, audio_dir="data/raw/audio", sample_rate=22050, duration=3.0, n_mels=128):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)  # 22050 * 3 = 66150 samples
        self.n_mels = n_mels
        
        self.classes = sorted([d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            folder_path = os.path.join(audio_dir, cls_name)
            for file_path in glob.glob(os.path.join(folder_path, "*.mp3")):
                self.samples.append((file_path, self.class_to_idx[cls_name]))
                
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def _process_waveform(self, waveform, sr):
        """Processes the audio waveform into a standardized Mel spectrogram."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        mel_spec = self.mel_spectrogram(waveform)
        return self.amplitude_to_db(mel_spec)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            waveform, sr = torchaudio.load(audio_path, backend="soundfile")
            return self._process_waveform(waveform, sr), label
        except Exception as e:
            print(f"Warning: Could not load {audio_path}. Returning zero tensor.")
            dummy_spec = torch.zeros((1, self.n_mels, 130))
            return dummy_spec, label


if __name__ == "__main__":
    dataset = BirdAudioDataset()
    print(f"Found classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"Total audio files: {len(dataset)}")
    
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Spectrogram tensor shape (C x Mels x Time): {spec.shape}")
        print(f"Label ID: {label}")