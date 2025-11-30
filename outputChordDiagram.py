#!/usr/bin/env python3

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
        lyrics_indiv_lines = lyrics_synced.strip().split("\n")
        formatted_chords = formattingOutput.format_synced_chord_grid(centisecond_chord, timestamps)
        formatted_chords_lines = formatted_chords.split("\n")
        print("Printing Chords:")

        if (len(lyrics_indiv_lines) != len(formatted_chords_lines)):
            print("Length diff")
        else:
            for i in range(len(formatted_chords_lines)):
                print(formatted_chords_lines[i])
                print(lyrics_indiv_lines[i])

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

def return_only_chords(predicted_chords: list[Optional[str]])->str:
    ''' Return only chords in order'''
    output = None
    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    output = formattingOutput.format_chord_grid(centisecond_chord)
    
    return output

def return_output_synced(lyrics_data: Dict[str, Any], predicted_chords: list[Optional[str]])->Optional[str]:
    ''' Return synced lyrics with chords '''
    output = ""

    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    lyrics_synced = lyrics_data.get("synced_lyrics")
    if lyrics_synced is None:
        pass
        #print("Error outputting chord synced lyrics.\nSynced lyrics not found")
    else:
        timestamps = formattingOutput.get_lyrics_timestamps(lyrics_synced)
        lyrics_indiv_lines = lyrics_synced.strip().split("\n")
        formatted_chords = formattingOutput.format_synced_chord_grid(centisecond_chord, timestamps)
        formatted_chords_lines = formatted_chords.split("\n")
        print("Printing Chords:")

        if (len(lyrics_indiv_lines) != len(formatted_chords_lines)):
            print("Length diff")
        else:
            for i in range(len(formatted_chords_lines)):
                output += formatted_chords_lines[i] + "\n" + lyrics_indiv_lines[i] + "\n"
                #print(formatted_chords_lines[i])
                #print(lyrics_indiv_lines[i])
    
    return output

def return_output_unsynced(lyrics_data: Dict[str, Any], predicted_chords: list[Optional[str]])->str:
    ''' Return chords and then lyrics separate '''
    output = ""

    print("As the lyrics found do not have timestamps, chords will be printed in order followed by the lyrics:\n")
    centisecond_chord = formattingOutput.chords_to_centiseconds(predicted_chords)
    formatted_output = formattingOutput.format_chord_grid(centisecond_chord)
    lyrics_plain = lyrics_data.get("plainLyrics")
    
    if lyrics_plain:
        output += formatted_output + "\n" + lyrics_plain
    
    '''print("Printing Chords:")
    print(formatted_output)
    print("Printing Lyrics:")
    print(lyrics_plain)'''

    return output