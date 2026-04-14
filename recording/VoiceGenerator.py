
import torchaudio
import torch
import soundfile as sf

# Monkey-patch torchaudio.load and torchaudio.save to use soundfile instead of torchcodec
# since torchaudio 2.11+ restricts backend routing and might fail due to torchcodec DLL issues on Windows.
def _soundfile_load(filepath, **kwargs):
    audio, sr = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(audio)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

def _soundfile_save(filepath, tensor, sample_rate, **kwargs):
    if tensor.ndim == 2:
        tensor = tensor.transpose(0, 1)
    sf.write(filepath, tensor.numpy(), sample_rate)

torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save

from f5_tts.api import F5TTS
import sounddevice as sd

class VoiceGenerator:
    
    def __init__(self):
        print("Loading F5-TTS model... (This may take a moment on first run)")
        self.tts = F5TTS() 
        
        self.ref_audio = r"C:\Users\Leon\Desktop\TUM\Master AI\Master Thesis\Riddle-Generator-RL\voicelines\scooty_scott_uncleaned.wav"
        
        self.ref_text = "Laddy, I was drinking scotch a hundred years before you were born, and I can tell you that whatever this is, it is definitely not scotch."

    def speak_custom(self, text):
        audio_data, sample_rate, _ = self.tts.infer(
            ref_file=self.ref_audio,
            ref_text=self.ref_text,
            gen_text=text
        )
        
        sd.play(audio_data, sample_rate)
        
        self.save_audio(audio_data, sample_rate, "voice_response.wav")

    def save_audio(self, audio_data, sample_rate, filename):
        """Saves audio data whether it's a Torch Tensor or a NumPy Array."""
        try:
            # 1. Convert to Tensor if it's currently a NumPy array
            if isinstance(audio_data, torch.Tensor):
                audio_to_save = audio_data.cpu()
            else:
                # It's already a numpy array, convert to tensor for torchaudio
                audio_to_save = torch.from_numpy(audio_data)
            
            # 2. Ensure the shape is (channels, frames) for torchaudio
            if audio_to_save.ndim == 1:
                audio_to_save = audio_to_save.unsqueeze(0)
            
            # 3. Save
            torchaudio.save(filename, audio_to_save, sample_rate)
            print(f"✅ Scotty's voice saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")