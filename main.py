#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
from numpy.linalg import norm
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import scipy

# Define chord templates

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_extended_chords():
    chords = {}
    
    # Major and minor
    major_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    minor_template = np.array([1.25, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    
    '''
    # Add 7th chords
    dom7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])  # Dominant 7th
    maj7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])  # Major 7th
    min7_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])  # Minor 7th
    '''

    for i in range(12):
        note = note_names[i]
        chords[note] = np.roll(major_template, i)
        chords[note + 'm'] = np.roll(minor_template, i)
        '''
        chords[note + '7'] = np.roll(dom7_template, i)
        chords[note + 'maj7'] = np.roll(maj7_template, i)
        chords[note + 'm7'] = np.roll(min7_template, i)'''
    
    return chords

def chord_template()->dict:
    ''' Generates a dictionary of chord templates
    key: Chord name e.g A or Am
    value: NDArray with 1 at each note in chord
    '''
    all_chords = generate_extended_chords()
    return all_chords

def chroma_func(myrecording: np.ndarray, frequency: int)->np.ndarray:
    ''' Constant Q transform after some processing
        smoothing at 2 seconds according to paper by Jiang et al Chroma features for chord recognition
    '''
    # isolate harmonics component
    recording_harm = librosa.effects.harmonic(y=myrecording, margin=8)

    # non-local filtering (computationally too expensive)
    '''chroma_filter = np.minimum(recording_harm,
                           librosa.decompose.nn_filter(recording_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))'''
    
    # smoothing at 2 seconds
    hop_len = 512

    chroma_cqt = librosa.feature.chroma_cens(
        y=recording_harm, sr=frequency, cqt_mode='hybrid', 
        hop_length=hop_len, win_len_smooth=2) #fmin=librosa.note_to_hz('C2')) will try this temporarily made results worse
    return chroma_cqt

def predict_chords(chroma_cqt: np.ndarray, all_chords: dict, threshold: float=0.825)->list:
    ''' Normalizes the chroma vector before going through all chord templates to find the best match
        normalizes it to get the relative differences in chroma vectors for a more accurate prediction
    '''
    predicted_chords = []

    for i in range(chroma_cqt.shape[1]):
        chroma_vector = chroma_cqt[:, i]

        # skip low energy
        if np.sum(chroma_vector) < 0.2:
            predicted_chords.append(None)
            continue

        best_match = None
        best_score = -1

        chroma_norm = norm(chroma_vector)
        if chroma_norm > 0:
            for chord_name, chord_template in all_chords.items():
                template_norm = norm(chord_template)
                if template_norm > 0:
                    score = np.dot(chroma_vector, chord_template) / (chroma_norm * template_norm)

                    if score > best_score:
                        best_score = score
                        best_match = chord_name
        else:
            predicted_chords.append(None)
        if best_score >= threshold:
            predicted_chords.append(best_match)
        else:
            predicted_chords.append(None)

    return predicted_chords

def post_process_chords(predicted_chords: list, min_frames: int|None) -> list:
    """
    Post-processing to remove very short chord segments
    min_duration: minimum number of frames a chord should last
    """
    if min_frames is None:
        min_frames = 3
    if not predicted_chords:
        return predicted_chords
    
    processed_chords = predicted_chords.copy()

    for i in range(1, len(processed_chords) - 1):
        # Remove isolated chord detections
        if (processed_chords[i] is not None and 
            processed_chords[i-1] != processed_chords[i] and 
            processed_chords[i+1] != processed_chords[i]):
            processed_chords[i] = None
        # Fill short gaps between same chords 
        elif (processed_chords[i] is None and 
            processed_chords[i-1] == processed_chords[i+1] and 
            processed_chords[i-1] is not None):
            processed_chords[i] = processed_chords[i-1]

    for i in range(min_frames, len(processed_chords) - 1 - min_frames):
        # Removes isolated chord detections
        if processed_chords[i] is not None:
            consecutive_frames = 0
            for x in range(min_frames):
                if processed_chords[i-x] is not None:
                    if processed_chords[i-x] == processed_chords[i]:
                        consecutive_frames += 1
                if processed_chords[i+x] is not None:
                    if processed_chords[i+x] == processed_chords[i]:
                        consecutive_frames += 1
            if consecutive_frames < min_frames:
                processed_chords[i] = None
    ''' WIP
    for i in range(min_frames, len(processed_chords) - 1 - min_frames):
        # fill empty chords with similar neighbors
        if processed_chords[i] is None:
            consecutive_frames = 0
            for x in range(min_frames):
                if processed_chords[i-x] is not None:
                    if processed_chords[i-x] == processed_chords[i]:
                        consecutive_frames += 1
                if processed_chords[i+x] is not None:
                    if processed_chords[i+x] == processed_chords[i]:
                        consecutive_frames += 1
            if consecutive_frames > min_frames:
                processed_chords[i] = 
    '''

    return processed_chords

def show_chords(predicted_chords: list[str]):
    chord = predicted_chords[0]
    prev_chord = predicted_chords[0]
    if chord is not None:
        print(chord)

    chord_list = []
    for i in range(5, len(predicted_chords)):
        chord = predicted_chords[i]
        if chord is None:
            continue

        if chord != prev_chord:
            chord_list.append(chord)
            prev_chord = chord

    chord_list = set(chord_list)
    print(" ".join(chord_list))

def show_graph(chroma_cqt: np.ndarray):
    # show graph
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram (CQT-based)')
    plt.show()

    print(chroma_cqt)

def main():
    ''' 
    parse input 
    input: -r is for recorded, -u upload audio
    '''
    ''' parse arguments '''
    # default
    Duration = 10
    frequency = 22050

    args = sys.argv[1:]

    if not args:
        print("Usage: -r [duration] or -u")
        sys.exit(1)

    flag = args[0]

    if flag == "-r":
        if len(args) > 1:
            try:
                Duration = float(args[1])
            except ValueError:
                print("Invalid duration, using default 10s.")
        print("Recording...")
        myrecording = sd.rec(int(Duration * frequency), samplerate=frequency, channels=1)
        sd.wait()
        myrecording = myrecording.flatten()

    elif flag == "-f":
    # For testing - accepts file path
        if len(args) < 2:
            print("Please provide a file path")
            sys.exit(1)
        path = args[1]
        myrecording, sr = librosa.load(path, sr=frequency, mono=True)
        myrecording = myrecording.flatten()

    else:
        print("Invalid Input")
        print("Usage: -r [duration] or -f [path]")
        print("-r to record audio, -f to upload file")
        sys.exit(1)

    all_chords = chord_template()
    chroma_cqt = chroma_func(myrecording, frequency)
    predicted_chords = predict_chords(chroma_cqt, all_chords)
    processed_chords = post_process_chords(predicted_chords, 5)
    #show_graph(chroma_cqt)
    show_chords(processed_chords)

if __name__ == '__main__':
    main()
