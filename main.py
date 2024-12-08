import os
import wave
import numpy as np

def load_audio_files(directory):
    audio_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            with wave.open(filepath, 'rb') as wave_file:
                frames = wave_file.readframes(wave_file.getnframes())
                signal = np.frombuffer(frames, dtype=np.int16)
                audio_data[filename] = signal
    return audio_data


if __name__ == '__main__':
    # directories
    motclefs_dir = 'data/motclefs'
    mots_inconnus_dir = 'data/mots_inconnus'

    # load audio files
    motclefs_audio = load_audio_files(motclefs_dir)
    mots_inconnus_audio = load_audio_files(mots_inconnus_dir)

    print("Mots clés chargés :", list(motclefs_audio.keys()))
    print("Mots inconnus chargés :", list(mots_inconnus_audio.keys()))