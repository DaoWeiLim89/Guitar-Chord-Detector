#!/usr/bin/env python3

import subprocess
import sys

# Expected chords for each song
expected_chords = {
    "About You": ["D", "A", "Bm", "G"],
    "All I Think About Now": ["D#", "Gm", "G#", "A#", "Cm", "C#", "B", "D", "C", "G", "F#", "F"],
    "Are You Looking Up": ["F#", "B", "C#sus4", "C#", "G#m", "E"], 
    "Back To Me": ["D", "Em7", "Gm", "E7", "Bm", "A", "E", "G", "Em"], 
    "Bad Habit": ["Cm", "C", "C#", "G#", "A#m", "D#", "Fm", "C7"],
    "Bags": ["F#", "A#m", "G#"],
    "Blue By You": ["B", "Bm", "C", "D", "E", "G", "G#"],
    "Creep": ["B", "C", "Cm", "G"],
    "Deceptacon": ["B", "A", "E", "D", "Bm", "G"],
    "Head In The Ceiling Fan": ["E", "A", "B", "G#m", "D#m"]
}

# Map songs to their audio files
audio_files = {
    "About You": "AudioFiles/About You.mp3",
    "All I Think About Now": "AudioFiles/All I Think About Now.mp3",
    "Are You Looking Up": "AudioFiles/Are You Looking Up.mp3",
    "Back To Me": "AudioFiles/Back To Me.mp3",
    "Bad Habit": "AudioFiles/Bad Habit.mp3",
    "Bags": "AudioFiles/Bags.mp3",
    "Blue By You": "AudioFiles/Blue by You.mp3",
    "Creep": "AudioFiles/Creep.mp3", 
    "Deceptacon": "AudioFiles/Deceptacon.mp3",
    "Head In The Ceiling Fan": "AudioFiles/Head In The Ceiling Fan.mp3",
}

def get_detected_chords(audio_file):
    """Run your chord detection program and get the chords"""
    try:
        # Run your program with the -f flag (you'll need to add this to your program)
        result = subprocess.run([
            sys.executable, "../main.py", "-f", audio_file
        ], capture_output=True, text=True)
        
        # Parse the output - assuming your program prints chords separated by spaces
        output = result.stdout.strip()
        if output:
            # Get the last line (where show_chords prints the results)
            lines = output.split('\n')
            chord_line = lines[-1] if lines else ""
            detected = chord_line.split()
            return detected
        return []
    except Exception as e:
        print(f"Error running program: {e}")
        return []

def calculate_accuracy(expected, detected):
    """Calculate what percentage of expected chords were found"""
    expected_set = set(expected)
    detected_set = set(detected)

    correct = len(expected_set.intersection(detected_set))
    total = len(expected_set)
    
    return (correct / total) * 100 if total > 0 else 0

def test_song(song_name):
    """Test a single song"""
    if song_name not in expected_chords:
        print(f"Song '{song_name}' not found")
        return
    
    audio_file = audio_files.get(song_name)
    if not audio_file:
        print(f"No audio file for '{song_name}'")
        return
    
    print(f"\n Testing: {song_name}")
    print("-" * 40)
    
    expected = expected_chords[song_name]
    detected = get_detected_chords(audio_file)
    expected_set = set(expected)
    detected_set = set(detected)
    difference = expected_set ^ detected_set

    accuracy = calculate_accuracy(expected, detected)
    
    print(f"Expected: {' '.join(expected)}")
    print(f"Detected: {' '.join(detected)}")
    print(f"Difference: {' '.join(difference)}")
    print(f"Accuracy: {accuracy:.1f}%")

def test_all_songs():
    """Test all songs"""
    total_accuracy = 0
    song_count = 0
    
    for song_name in expected_chords.keys():
        if song_name in audio_files:
            test_song(song_name)
            
            expected = expected_chords[song_name]
            detected = get_detected_chords(audio_files[song_name])
            accuracy = calculate_accuracy(expected, detected)
            
            total_accuracy += accuracy
            song_count += 1
    
    if song_count > 0:
        avg_accuracy = total_accuracy / song_count
        print(f"\n📊 Overall Average Accuracy: {avg_accuracy:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific song
        test_song(sys.argv[1])
    else:
        # Test all songs
        test_all_songs()
