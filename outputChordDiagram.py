#!/usr/bin/env python3

import lyrics
import numpy as np
import librosa
from typing import Optional, Dict, Any
import formattingOutput

def display_only_chords(predicted_chords: list[Optional[str]])->None:
    ''' Displays only chords in order'''
    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    formatted_output = formattingOutput.format_chord_grid(centisecond_chord)
    print(formatted_output)

def display_output_synced(lyrics_data: Dict[str, Any], predicted_chords: list[Optional[str]])->None:
    ''' Displays synced lyrics with chords '''
    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    lyrics_synced = lyrics_data.get("synced_lyrics")
    if lyrics_synced is None:
        print("Error outputting chord synced lyrics.\nSynced lyrics not found")
    else:
        timestamps = formattingOutput.get_lyrics_timestamps(lyrics_synced)
        lyrics_indiv_lines = lyrics_synced.split("\n") # works, need to iterate through
        formatted_chords = formattingOutput.format_synced_chord_grid(centisecond_chord, timestamps) # change this
        print("Printing Chords:")
        #print(formatted_chords)
        print(formatted_chords)


def display_output_unsynced(lyrics_data: Dict[str, Any], predicted_chords: list[Optional[str]])->None:
    ''' Displays chords and then lyrics separate '''
    print("As the lyrics found do not have timestamps, chords will be printed in order followed by the lyrics:\n")
    
    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    formatted_output = formattingOutput.format_chord_grid(centisecond_chord)
    lyrics_plain = lyrics_data.get("plainLyrics")
    print("Printing Chords:")
    print(formatted_output)
    print("Printing Lyrics:")
    print(lyrics_plain)