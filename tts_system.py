from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
import requests
import os
import io
import re

# --- Patch to allow picke loading of Kokoro instances ---
try:
    _original_load = np.load
    # Define a wrapper that defaults allow_pickle to True
    def _patched_load(*args, **kwargs):
        if 'allow_pickle' not in kwargs:
            kwargs['allow_pickle'] = True
        return _original_load(*args, **kwargs)
    # Apply the patch
    np.load = _patched_load
except Exception as e:
    print(f"Warning: Could not patch NumPy: {e}")
    


class KokoroTTS:

    CLEAN_REGEX = re.compile(r'[^\w\s.,!?;:\'\-\"()$]')


    def __init__(self):
        """
        Initializes the Kokoro TTS engine.
        Ensures model weights are present before loading the ONNX session.
        """

        self.model_path = "kokoro-v0_19.onnx"
        self.voices_path = "voices.bin"
        self._ensure_models_exist()

        # Initialize the model instance
        self.kokoro = Kokoro(self.model_path, self.voices_path)
    
    def _ensure_models_exist(self):
        """
        Downloads the model weights and voice configs if they are missing.
        Uses standard requests without UI spinners to keep this class pure.
        """

        # URLs for the ONNX model and voices file (v0.19)
        MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

        if not os.path.exists(self.model_path):
            print("Downloading Kokoro Model (this may take a moment)...")
            response = requests.get(MODEL_URL, stream=True)

            # Save the model file in chunks to avoid memory issues
            with open(self.model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model Downloaded.")


        if not os.path.exists(self.voices_path):
            print("Downloading Voices Config...")
            response = requests.get(VOICES_URL)

             # Save the voices file
            with open(self.voices_path, "wb") as f:
                f.write(response.content)
            print("Voices Config Downloaded.")


    def _check_text(self,text: str) -> tuple[bool,str]:
        """
        Docstring for _check_text
        
        :param text: Description
        :type text: str
        """
        if text is None or not isinstance(text,str) or text.strip() == "":
            return False, ''
        
        # Normalize whitespace
        text = " ".join(text.split())
        text = self.CLEAN_REGEX.sub('', text)

        if len(text.split()) > 250:
            print("Warning: Text exceeds 500 characters. Truncating to fit model limits.")
            text = " ".join(text.split()[:250])
            end_space = text.rfind(' ')
            if end_space > 0:
                text = text[:end_space] + "..."

        # For Debugging: Uncomment the following lines to see the cleaned text before generation
        # print("\n --- Text Check Passed ---")
        # print(text)
        # print("--- End of Text Check ---\n")
        return True, text
        


    def generate_audio(self, text: str, voice: str = "af_sarah", speed: float = 1.0):
        """
        Generates audio bytes from text.
        Returns None if text is empty.
        """
        check_result, text = self._check_text(text)
        if not check_result:
            print("Warning: Empty or invalid text provided. Returning empty audio.")
            return None
        
        # Generate audio samples using the Kekoro model
        samples, sample_rate = self.kokoro.create(
                text, 
                voice=voice, 
                speed=speed, 
                lang="en-us"
            )
        
        # Convert the samples to WAV format in memory
        byte_io = io.BytesIO()
        sf.write(byte_io, samples, sample_rate, format='WAV')
        byte_io.seek(0)
        return byte_io


if __name__ == "__main__":
    # Test the KokoroTTS class by generating audio from a sample text
    
    sample = KokoroTTS()
    sample_text = """
    Here is your accessibility guide for nearby supermarkets. I found nine locations in your vicinity.
    For confirmed accessibility, Waitrose in Downtown Dubai, Spinneys in DIFC, and both Carrefour Markets at Marina Crown and Silverene Towers offer fully accessible entrances and restrooms. The West Zone Fresh Hypermarket near the metro station in Al Barsha also provides full accessibility.
    Several other locations, including Aswaaq in Al Sofouh, Al Maya, and VIVA Supermarket, have confirmed accessible entrances, but their restroom accessibility is currently unknown. Please note that the 'Gate 1 to 20' location in Al Barsha has an accessible entrance but does not have an accessible restroom.
    The closest option with full verified accessibility is Waitrose, which is 1.2 kilometers away."
"""
    audio = sample.generate_audio(sample_text)
    if audio is not None:
        with open("sample_output.wav", "wb") as f:
            f.write(audio.read())

