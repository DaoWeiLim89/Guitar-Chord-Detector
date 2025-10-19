#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
from numpy.linalg import norm
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import scipy
import lyrics
import chromaGeneration

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
    # Records Audio to Anaylze
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
    # Accepts file path (MP3)
        if len(args) < 2:
            print("Please provide a file path")
            sys.exit(1)
        path = args[1]
        myrecording, sr = librosa.load(path, sr=frequency, mono=True)
        myrecording = myrecording.flatten()

    elif flag == "-fl":
    # Accepts file path and prints with lyrics
        if not (len(args) == 3 or len(args) == 4):
            print("Incorrect usage: -lf [Filepath] [Song Name] [Artist Name (Optional)]")
            sys.exit(1)

        path = args[1]
        songName = args[2]
        if args[3]:
            artistName = args[3]

        myrecording, sr = librosa.load(path, sr=frequency, mono=True)
        myrecording = myrecording.flatten()
        songLyrics = lyrics.get_lyrics(songName, artistName)

    else:
        print("Invalid Input")
        print("Usage: -r [duration] or -f [path] or -fl [path] [Song Name] [Artist Name (Optional)]")
        print("-r to record audio, -f to upload file, -fl to upload file and get lyrics")
        sys.exit(1)

    all_chords = chromaGeneration.chord_template()
    chroma_cqt = chromaGeneration.chroma_func(myrecording, frequency)
    predicted_chords = chromaGeneration.predict_chords(chroma_cqt, all_chords)
    processed_chords = chromaGeneration.post_process_chords(predicted_chords, 5)
    #show_graph(chroma_cqt)
    show_chords(processed_chords)

if __name__ == '__main__':
    main()
