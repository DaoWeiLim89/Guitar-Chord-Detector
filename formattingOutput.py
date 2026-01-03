#!/usr/bin/env python3

import numpy as np
from typing import List, Optional

import numpy as np
from typing import List, Optional

def chords_to_centiseconds(processed_chords: List[Optional[str]], 
                          hop_length: int = 2048, 
                          sr: int = 11025) -> List[Optional[str]]:
    """
    Convert chord predictions from frame-based to centisecond-based timing.
    Args:
        processed_chords: List of chord predictions per frame (can contain None)
        hop_length: Hop length used in chroma_cqt (default 2048)
        sr: Sample rate (default 11025)
    
    Returns:
        List of chords where each index represents one centisecond (0.01s)
    """
    # Calculate num of centiseconds per frame
    cs_per_frame = (hop_length / sr) * 100
    
    # Calculate total duration
    total_cs = int(np.ceil(len(processed_chords) * cs_per_frame))

    centisecond_chords = [None] * total_cs
    
    for cs in range(total_cs):
        # Using integer math here is slightly faster and safer
        frame_idx = int(cs / cs_per_frame)
        
        if frame_idx < len(processed_chords):
            centisecond_chords[cs] = processed_chords[frame_idx]
    
    return centisecond_chords
'''
def chords_to_centiseconds(processed_chords: List[Optional[str]], hop_length: int = 512, sr: int = 22050) -> List[Optional[str]]:
    """
    Convert chord predictions from frame-based to centisecond-based timing.
    Args:
        processed_chords: List of chord predictions per frame (can contain None)
        hop_length: Hop length used in chroma_cqt (default 512)
        sr: Sample rate (default 22050)
    Returns: List of chords where each index represents one centisecond (0.01s)
    """
    # Calculate time per frame in seconds
    time_per_frame = hop_length / sr

    # Calculate total duration in seconds
    total_duration = len(processed_chords) * time_per_frame

    # Calculate number of centiseconds
    num_centiseconds = int(np.ceil(total_duration * 100))

    # Initialize output array with explicit type hint
    centisecond_chords: List[Optional[str]] = [None] * num_centiseconds

    # Map each centisecond to the corresponding frame
    for cs in range(num_centiseconds):
        # Convert centisecond to seconds
        time_in_seconds = cs / 100.0

        # Find corresponding frame
        frame_idx = int(time_in_seconds / time_per_frame)

        # Ensure we don't go out of bounds
        if frame_idx < len(processed_chords):
            centisecond_chords[cs] = processed_chords[frame_idx]

    return centisecond_chords 
'''

def format_lrc_timestamp(centiseconds: int) -> str:
    """
    Format centiseconds as LRC timestamp [MM:SS.CC]
    Args:
        centiseconds: Time in centiseconds
    Returns:
        Formatted timestamp string like [00:12.34]
    """
    total_seconds = centiseconds / 100.0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    cents = int((total_seconds % 1) * 100)
    
    return f"[{minutes:02d}:{seconds:02d}.{cents:02d}]"

def timestamp_to_cs(timestamp: str)->int:
    timestamp = timestamp[1:-1]
    mins, cseconds = timestamp.split(":")
    centiseconds = int(float(mins) * 60 * 100 + float(cseconds) * 100)
    return centiseconds

def get_chord_changes(centisecond_chords: List[Optional[str]]) -> List[tuple]:
    """
    Extract chord changes with timestamps.
    Returns list of (centisecond, chord) tuples only when chord changes.
    Args:
        centisecond_chords: List of chords per centisecond
    Returns:
        List of (centisecond, chord_name) tuples
    """
    changes = []
    current_chord = None
    
    for cs, chord in enumerate(centisecond_chords):
        if chord != current_chord:
            changes.append((cs, chord))
            current_chord = chord
    
    return changes

def format_chord_timeline(centisecond_chords: List[Optional[str]], 
                         show_all: bool = False) -> str:
    """
    Format chords as a readable timeline with LRC timestamps
    Args:
        centisecond_chords: List of chords per centisecond
        show_all: If True, show every centisecond. If False, show only changes.
    Returns:
        Formatted string with timestamps and chords
    """
    output = []
    
    if show_all:
        # Show every centisecond (very verbose, usually not recommended)
        for cs, chord in enumerate(centisecond_chords):
            timestamp = format_lrc_timestamp(cs)
            chord_name = chord if chord is not None else "No chord"
            output.append(f"{timestamp} {chord_name}")
    else:
        # Show only chord changes (recommended)
        changes = get_chord_changes(centisecond_chords)
        for cs, chord in changes:
            timestamp = format_lrc_timestamp(cs)
            chord_name = chord if chord is not None else "No chord"
            output.append(f"{timestamp} {chord_name}")
    
    return "\n".join(output)

def format_chord_grid(centisecond_chords: List[Optional[str]], 
                     interval_seconds: int = 5) -> str:
    """
    Format chords in a grid showing chord changes within time intervals.
    Consecutive duplicate chords are not shown, and spacing reflects when changes occur.
    Args:
        centisecond_chords: List of chords per centisecond
        interval_seconds: Time interval in seconds for each row (default 5)
    Returns:
        Formatted string with grid layout
    Example:
        [00:05.00] C       D              E
        [00:10.00] C        D              E
        [00:15.00] C   G             E
    """
    output = []
    interval_cs = interval_seconds * 100  # Convert to centiseconds
    total_duration_cs = len(centisecond_chords)
    
    # Process each interval
    start_cs = 0
    while start_cs < total_duration_cs:
        end_cs = min(start_cs + interval_cs, total_duration_cs)
        
        # Get timestamp for this interval
        timestamp = format_lrc_timestamp(end_cs)
        
        # Collect chord changes within this interval
        changes_in_interval = []
        last_chord = None
        
        for cs in range(start_cs, end_cs):
            current_chord = centisecond_chords[cs]
            
            # Only record when chord changes and skip None
            if current_chord != last_chord and current_chord is not None:
                # Calculate relative position within interval (0 to 1)
                relative_position = (cs - start_cs) / interval_cs
                changes_in_interval.append((relative_position, current_chord))
                last_chord = current_chord
        
        # Format the line with spacing
        if changes_in_interval:
            line = timestamp + " "
            
            # Create spacing based on relative positions
            # Use a fixed character width for the chord display area
            chord_area_width = 60
            
            for i, (position, chord) in enumerate(changes_in_interval):
                # Calculate spacing before this chord
                if i == 0:
                    # First chord - add spaces based on its position
                    num_spaces = int(position * chord_area_width)
                    line += " " * num_spaces
                else:
                    # Subsequent chords - space based on distance from previous
                    prev_position = changes_in_interval[i-1][0]
                    prev_chord = changes_in_interval[i-1][1]
                    
                    # Distance between chords
                    distance = position - prev_position
                    num_spaces = int(distance * chord_area_width) - len(prev_chord)
                    num_spaces = max(1, num_spaces)  # At least 1 space
                    line += " " * num_spaces
                
                line += chord
            
            output.append(line)
        else:
            # No chords in this interval
            output.append(timestamp)
        
        start_cs = end_cs
    
    return "\n".join(output)

def get_lyrics_timestamps(synced_lyrics: str)->list[str]:
    ''' Extracts time stamps from synced_lyrics '''
    timestamps = []
    lines = synced_lyrics.split('\n')
    for line in lines:
        timestamps.append(line[0:10])
    return timestamps

def format_synced_chord_grid(centisecond_chords: List[Optional[str]], timestamps: list[str]) -> str:
    """
    Format chords in a grid showing chord changes within time intervals specified by timestamps.
    Consecutive duplicate chords are not shown, and spacing reflects when changes occur.
    Args:
        centisecond_chords: List of chords per centisecond
        timestamps: List of strings containing each time interval
    Returns:
        Formatted string with grid layout
    Example:
        [00:05.53] C       D              E
        [00:13.28] C        D              E
        [00:17.11] C   G             E
    """
    output = []
    total_duration_cs = len(centisecond_chords)
    total_duration_lyrics = timestamp_to_cs(timestamps[-1])
    if total_duration_lyrics > total_duration_cs:
        print("Lyrics longer than duration of mp3 file. Likely lyrics mismatch. Exiting")
        exit(1)
    start_interval_cs = 0
    prev_timestamp = "[00:00.00]"
    # Process each interval
    for timestamp in timestamps[1:]:
        end_interval_cs = timestamp_to_cs(timestamp)

        # Collect chord changes within this interval
        changes_in_interval = []
        last_chord = None

        for cs in range(start_interval_cs, end_interval_cs):
            current_chord = centisecond_chords[cs]

            # Only record when chord changes and skip None
            if current_chord != last_chord and current_chord is not None:
                # Calculate relative position within interval (0 to 1)
                relative_position = (cs - start_interval_cs) / (end_interval_cs-start_interval_cs)
                changes_in_interval.append((relative_position, current_chord))
                last_chord = current_chord
        
        # Format the line with spacing
        if changes_in_interval:
            line = prev_timestamp + " "
            
            # Create spacing based on relative positions
            # Use a dynamic character width for the chord display area
            chord_area_width = int(50 / 10 * ((end_interval_cs - start_interval_cs)/100)) #assuming 10s = max width
            
            for i, (position, chord) in enumerate(changes_in_interval):
                # Calculate spacing before this chord
                if i == 0:
                    # First chord - add spaces based on its position
                    num_spaces = int(position * chord_area_width)
                    line += " " * num_spaces
                else:
                    # Subsequent chords - space based on distance from previous
                    prev_position = changes_in_interval[i-1][0]
                    prev_chord = changes_in_interval[i-1][1]
                    
                    # Distance between chords
                    distance = position - prev_position
                    num_spaces = int(distance * chord_area_width) - len(prev_chord)
                    num_spaces = max(1, num_spaces)  # At least 1 space
                    line += " " * num_spaces
                
                line += chord
            output.append(line)
        else:
            # No chords in this interval
            output.append(prev_timestamp)
        
        start_interval_cs = end_interval_cs
        prev_timestamp = timestamp

    # last timestamp to end of song segment
    output.append(prev_timestamp + " End")
    return "\n".join(output)

# Example usage
if __name__ == "__main__":
    # Simulate some processed chords
    example_processed_chords = ['C'] * 10 + ['Am'] * 15 + [None] * 5 + ['F'] * 12 + ['G'] * 10
    
    # Convert to centiseconds
    centisecond_chords = chords_to_centiseconds(
        example_processed_chords, 
        hop_length=2048, 
        sr=11025
    )
    
    print("Centisecond-based chords (first 50):")
    print(centisecond_chords[:50])
    print(f"\nTotal centiseconds: {len(centisecond_chords)}")
    
    # Get chord changes
    changes = get_chord_changes(centisecond_chords)
    print(f"\nChord changes: {len(changes)}")
    
    # Format as timeline
    print("\nChord Timeline (changes only):")
    print(format_chord_timeline(centisecond_chords, show_all=False))
    
    # You can also access specific centisecond
    centisecond_1234 = 1234  # 12.34 seconds
    if centisecond_1234 < len(centisecond_chords):
        print(f"\nChord at [00:12.34]: {centisecond_chords[centisecond_1234]}")