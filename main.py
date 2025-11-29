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
import outputChordDiagram
from typing import Optional, Dict, Any

def show_chords(predicted_chords: list[Optional[str]]):
    chord = predicted_chords[0]
    prev_chord = predicted_chords[0]
    chord_list = []
    if chord is not None:
        chord_list.append(chord)

    for i in range(5, len(predicted_chords)):
        chord = predicted_chords[i]
        if chord is None:
            continue

        if chord != prev_chord:
            chord_list.append(chord)
            prev_chord = chord

    #chord_list = set(chord_list)
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
    all_chords = chromaGeneration.chord_template()

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
        chroma_cqt = chromaGeneration.chroma_func(myrecording, frequency)
        predicted_chords = chromaGeneration.predict_chords(chroma_cqt, all_chords)
        processed_chords = chromaGeneration.post_process_chords(predicted_chords, 5)
        #show_graph(chroma_cqt)
        show_chords(processed_chords)

    elif flag == "-f":
    # Accepts file path (MP3)
        if len(args) < 2:
            print("Please provide a file path")
            sys.exit(1)
        path = args[1]
        myrecording, sr = librosa.load(path, sr=frequency, mono=True)
        myrecording = myrecording.flatten()
        chroma_cqt = chromaGeneration.chroma_func(myrecording, frequency)
        predicted_chords = chromaGeneration.predict_chords(chroma_cqt, all_chords)
        processed_chords = chromaGeneration.post_process_chords(predicted_chords, 5)
        #show_graph(chroma_cqt)
        show_chords(processed_chords)

    elif flag == "-fl":
    # Accepts file path and prints with lyrics
        if not (len(args) == 3 or len(args) == 4):
            print("Incorrect usage: -lf [Filepath] [Song Name] [Artist Name (Optional)]")
            sys.exit(1)

        path = args[1]
        songName = args[2]
        artistName = ""
        if len(args) == 4:
            artistName = args[3]

        # Get file recording
        myrecording, sr = librosa.load(path, sr=frequency, mono=True)
        myrecording = myrecording.flatten()
        # Process Chords
        chroma_cqt = chromaGeneration.chroma_func(myrecording, frequency)
        predicted_chords = chromaGeneration.predict_chords(chroma_cqt, all_chords)
        processed_chords = chromaGeneration.post_process_chords(predicted_chords, 5)
        #show_graph(chroma_cqt)
        #show_chords(processed_chords)

        songLyrics = lyrics.get_lyrics(songName, artistName)
        isSynced = False
        synced_lyrics = None
        unsynced_lyrics = None

        if songLyrics is not None:
            # Process Lyrics
            songLyrics = lyrics.prepend_lyrics(songLyrics) # prepend with [00:00.00]
            synced_lyrics = lyrics.get_synced_lyrics(songLyrics)
            unsynced_lyrics = lyrics.get_unsynced_lyrics(songLyrics)

            if synced_lyrics is not None:
                isSynced = True

            if synced_lyrics is None and unsynced_lyrics is None:
                print("Lyrics could not be found\nPrinting Chords:")
                outputChordDiagram.display_only_chords(processed_chords)

            if isSynced:
                print("Now outputting synced lyrics")
                outputChordDiagram.display_output_synced(songLyrics, processed_chords)
            else:
                print("Now outputting unsynced lyrics")
                outputChordDiagram.display_output_unsynced(songLyrics, processed_chords)
            
        else:
            print("Lyrics could not be found\nPrinting Chords:")
            outputChordDiagram.display_only_chords(processed_chords)
        

    else:
        print("Invalid Input")
        print("Usage: -r [duration] or -f [path] or -fl [path] [Song Name] [Artist Name (Optional)]")
        print("-r to record audio, -f to upload file, -fl to upload file and get lyrics")
        sys.exit(1)

if __name__ == '__main__':
    main()
