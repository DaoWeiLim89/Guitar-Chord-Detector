#!/usr/bin/env python3

import numpy as np
import librosa
from numpy.linalg import norm
from typing import Optional, Dict, Any
import collections

# Define chord templates

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_extended_chords():
    chords = {}
    
    # Major and minor
    major_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    minor_template = np.array([1.25, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    # root note greatest prominence
    # template matching kind of sucks when comparing chords with 2/3 shared notes
    

    # Add 7th chords
    dom7_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.7, 0])  # Dominant 7th
    maj7_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.7])  # Major 7th
    min7_template = np.array([1.25, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.7, 0])  # Minor 7th


    for i in range(12):
        note = note_names[i]
        chords[note] = np.roll(major_template, i)
        chords[note + 'm'] = np.roll(minor_template, i)

        chords[note + '7'] = np.roll(dom7_template, i)
        chords[note + 'maj7'] = np.roll(maj7_template, i)
        chords[note + 'm7'] = np.roll(min7_template, i)
    
    return chords

def chord_template()->dict:
    ''' Generates a dictionary of chord templates
    key: Chord name e.g A or Am
    value: NDArray with 1 at each note in chord
    '''
    all_chords = generate_extended_chords()
    return all_chords

def chroma_func(myrecording: np.ndarray, frequency: int)->np.ndarray:
    '''
        win_len_smooth: number of frames for smoothing window
        n_chroma: number of chroma bins to produce -> 12 for 12 semitones
    '''
    # isolate harmonics component (strips away percussive elements)
    recording_harm = librosa.effects.harmonic(y=myrecording, margin=8)
    
    hop_len = 512
    win_len_smooth = int(0.2 * frequency / hop_len)  # 0.2s seconds window
    f_min = librosa.note_to_hz('C2')  # Set minimum frequency to C2

    # Hybrid approach with all 3
    cqt = librosa.feature.chroma_cqt(y=recording_harm, sr=frequency, hop_length=hop_len)
    stft = librosa.feature.chroma_stft(y=recording_harm, sr=frequency, hop_length=hop_len)
    cens = librosa.feature.chroma_cens(
        y=recording_harm, sr=frequency, cqt_mode='hybrid', 
        hop_length=hop_len, win_len_smooth=win_len_smooth, 
        fmin=f_min)

    # weights for each chroma feature
    cqt_w = 0.6
    stft_w = 0.3
    cens_w = 0.1

    combined = (cqt_w * cqt) + (stft_w * stft) + (cens_w * cens)

    # square values to increase contrast
    combined = np.power(combined, 2)

    # remove low energy notes
    for t in range(combined.shape[1]):
        col = combined[:, t]
        # Only keep notes > 10% of the strongest note
        threshold = 0.1 * np.max(col)
        combined[col < threshold, t] = 0

    chroma_cqt = librosa.util.normalize(combined, norm=2, axis=0) # norm=2 for Euclidean norm

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    
    for t in [0, 100, 500, 1000, 2000]:
        if t < chroma_cqt.shape[1]:
            col = chroma_cqt[:, t]
            top3_idx = np.argsort(col)[-3:][::-1]
            print(f"\nFrame {t}:")
            print(f"  All 12 bins: {col}")
            print(f"  Notes:       {note_names}")
            print(f"  Top 3: {[note_names[i] for i in top3_idx]} = {col[top3_idx]}")
            print(f"  Max/Min ratio: {col.max() / max(col.min(), 1e-6):.1f}")
    
    np.set_printoptions()
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

'''
def post_process_chords(predicted_chords: list, min_frames: int|None) -> list[Optional[str]]:
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

    return processed_chords
    '''
# implement the new post-processing method here

def post_process_chords(predicted_chords: list, window_size: int = 11) -> list:
    ''' Processes chords using a Categorical Median Filter to clean up noisy predictions 
    and fill small gaps with a majority vote
    Uses a sliding window approach. Window size = width of window
    '''
    if not predicted_chords:
        return predicted_chords
    
    processed_chords = []
    radius = window_size // 2

    for i in range(len(predicted_chords)):
        start = max(0, i-radius)
        end = min(len(predicted_chords), i+radius+1)
        window = predicted_chords[start:end]

        valid_chords = [chord for chord in window if chord is not None] # filter out None values

        if not valid_chords:
            processed_chords.append(None)
            continue
        
        counts = collections.Counter(valid_chords)
        most_common_chord, freq = counts.most_common(1)[0]

        # tie breaker - choose the current chord if tied
        cur_chord = predicted_chords[i]
        if cur_chord is not None and counts[cur_chord] == freq:
            processed_chords.append(cur_chord)
        else:
            processed_chords.append(most_common_chord)

    return processed_chords