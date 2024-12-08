import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from functions.extractionCoeffCepstraux import extract_mfcc_feature

def load_audio_files(directory):
    audio_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            with wave.open(filepath, 'rb') as wave_file:
                frames = wave_file.readframes(wave_file.getnframes())
                signal = np.frombuffer(frames, dtype=np.int16)
                framerate = wave_file.getframerate()
                audio_data[filename] = (signal, framerate)
    return audio_data

# on gere les fenetres glissantes
def buffer(signal, frame_size, frame_step):
    num_frames = (len(signal) - frame_size) // frame_step + 1
    frames = []
    for i in range(num_frames):
        start = i * frame_step
        frame = signal[start:start + frame_size]
        frames.append(frame)
    return np.array(frames)


# extraction des coeff avec la fonction fournit
def extract_coefficients(signal, framerate):
    frame_size = int(0.03 * framerate)  # Fenêtre de 30ms
    frame_step = int(0.015 * framerate)  # Pas de 15ms
    frames = buffer(signal, frame_size, frame_step)
    cepstraux = []
    for frame in frames:
        coeffs = extract_mfcc_feature(frame, framerate)
        cepstraux.append(coeffs)
    return np.array(cepstraux)

# DTW algorithm implementation (with AI help)
def dtw(ref_coeffs, test_coeffs):
    n = len(ref_coeffs)
    m = len(test_coeffs)
    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n):
        for j in range(1, m):
            cost = np.linalg.norm(ref_coeffs[i] - test_coeffs[j])  # Distance euclidienne
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # insert
                dtw_matrix[i, j - 1],     # delete
                dtw_matrix[i - 1, j - 1]  # match
            )

    total_cost = dtw_matrix[-1, -1]
    return total_cost

# identify the closest word
def decision(distances, dictionary):
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return dictionary[min_index]

# visualisation of distances
def visualisation(distances, dictionary, file_name):
    plt.bar(dictionary, distances)
    plt.xlabel("Words")
    plt.ylabel("Distance")
    plt.title("Comparaison des distances pour le fichier " + file_name)
    plt.show()

def affichage(mot):
    print("Recognized word " + mot)

if __name__ == "__main__":
    # directories
    motclefs_dir = 'data/motclefs'
    mots_inconnus_dir = 'data/mots_inconnus'

    # load audio files
    motclefs_audio = load_audio_files(motclefs_dir)
    mots_inconnus_audio = load_audio_files(mots_inconnus_dir)

    dictionary = list(motclefs_audio.keys())

    # extract les coefficients cepstraux
    reference_matrices = {}
    for filename, (signal, framerate) in motclefs_audio.items():
        reference_matrices[filename] = extract_coefficients(signal, framerate)

    # compare with unknown words
    for unknown_filename, (unknown_signal, unknown_framerate) in mots_inconnus_audio.items():
        # Extraire les coefficients cepstraux du mot inconnu
        test_coeffs = extract_coefficients(unknown_signal, unknown_framerate)

        # process distances
        distances = []
        for ref_filename, ref_coeffs in reference_matrices.items():
            distance = dtw(ref_coeffs, test_coeffs)
            distances.append(distance)

        # decision
        recognized_word = decision(distances, dictionary)

        # visualisation
        print("file :", unknown_filename)
        affichage(recognized_word)
        visualisation(distances, dictionary, unknown_filename)