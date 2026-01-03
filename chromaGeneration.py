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
    
    '''
    # Add 7th chords
    dom7_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.7, 0])  # Dominant 7th
    maj7_template = np.array([1.25, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.7])  # Major 7th
    min7_template = np.array([1.25, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.7, 0])  # Minor 7th
    '''
    # Normalize all templates to unit length
    major_template = major_template / norm(major_template)
    minor_template = minor_template / norm(minor_template)
    '''
    dom7_template = dom7_template / norm(dom7_template)
    maj7_template = maj7_template / norm(maj7_template)
    min7_template = min7_template / norm(min7_template)'''


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

    ''' Debugging info to see chroma values at certain frames
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
    
    np.set_printoptions()'''
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
            chroma_unit = chroma_vector / chroma_norm
            for chord_name, chord_template in all_chords.items():
                score = np.dot(chroma_unit, chord_template)

                if score > best_score:
                    best_score = score
                    best_match = chord_name
        else:
            predicted_chords.append(None)
            continue

        if best_score >= threshold:
            predicted_chords.append(best_match)
        else:
            predicted_chords.append(None)

    return predicted_chords
'''
def post_process_chords(predicted_chords: list, window_size: int = 15) -> list:
    Processes chords using a Categorical Median Filter to clean up noisy predictions 
    and fill small gaps with a majority vote
    Uses a sliding window approach. Window size = width of window

    
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
'''

def post_process_chords(predicted_chords: list, window_size: int = 15) -> list:
    ''' Processes chords using a Categorical Median Filter to clean up noisy predictions 
    and fill small gaps with a majority vote
    Uses a sliding window approach. Window size = width of window
    '''
    
    if not predicted_chords:
        return predicted_chords
    
    n = len(predicted_chords)
    processed_chords = []
    radius = window_size // 2
    most_frequent = collections.Counter()

    # Initialize window for i=0: indices [0, radius]
    for j in range(min(radius + 1, n)):
        if predicted_chords[j] is not None:
            most_frequent[predicted_chords[j]] += 1

    for i in range(n):
        # Find best chord from current window
        best_chord = None
        max_count = 0
        for chord, count in most_frequent.items():
            if count > max_count:
                max_count = count
                best_chord = chord

        # Tie-breaker: prefer current chord
        cur = predicted_chords[i]
        if cur is not None and most_frequent[cur] == max_count:
            best_chord = cur

        processed_chords.append(best_chord if max_count > 0 else None)

        # Slide window for next iteration
        left_out = i - radius
        right_in = i + radius + 1
        
        if left_out >= 0 and predicted_chords[left_out] is not None:
            most_frequent[predicted_chords[left_out]] -= 1
            if most_frequent[predicted_chords[left_out]] == 0:
                del most_frequent[predicted_chords[left_out]]
        if right_in < n and predicted_chords[right_in] is not None:
            most_frequent[predicted_chords[right_in]] += 1

    return processed_chords