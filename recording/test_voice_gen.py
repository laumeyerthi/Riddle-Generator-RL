import os
from pathlib import Path
import os
# Force torchaudio to use the stable soundfile backend instead of torchcodec
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

ffmpeg_path = r"C:\ffmpeg\bin"
if os.path.exists(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)
from VoiceGenerator import VoiceGenerator

def test():
    speaker = VoiceGenerator()
    speaker.speak_custom("I recommend moving up. This will take you closer to your goal, though you may need to press a button to fully open the way.")




if __name__ == "__main__":
    test()