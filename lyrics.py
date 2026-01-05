#!/usr/bin/env python3

import requests
import json
from typing import Optional, Dict, Any

def get_lyrics(song_name: str, artist_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Search for a song on lrclib.net and return the synced lyrics for the top result.
    
    Args:
        song_name (str): Name of the song to search for
        artist_name (str, optional): Artist name to improve search accuracy
        
    Returns:
        Dict containing song info and synced lyrics, or None if not found
    """
    
    # Base URL for lrclib API
    base_url = "https://lrclib.net/api"
    
    try:
        # Search for song
        search_url = f"{base_url}/search"
        search_params = {
            "q": f"{artist_name} {song_name}".strip()
        }
        
        print(f"Searching for: {search_params['q']}")
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        
        search_results = search_response.json()
        print(f"Search results: {len(search_results)} found")
        #print(search_results)
        
        if not search_results:
            print("No search results found")
            return None
        
        # Try finding synced lyrics
        top_result = search_results[0] # default to first result
        for result in search_results:
            if result.get('syncedLyrics'):
                top_result = result
                break
        
        song_id = top_result.get('id')
        
        if not song_id:
            print("No valid song ID found in top result")
            return None
        
        print(f"Found song: {top_result.get('name', 'Unknown')} by {top_result.get('artistName', 'Unknown Artist')}")
        
        # Get detailed info including synced lyrics
        detail_url = f"{base_url}/get/{song_id}"
        detail_response = requests.get(detail_url)
        detail_response.raise_for_status()
        
        song_details = detail_response.json()
        
        # Check if synced lyrics are available
        if song_details.get('syncedLyrics'):
            print("Synced lyrics found on lrclib!")
            return {
                'id': song_details.get('id'),
                'name': song_details.get('name'),
                'artist': song_details.get('artistName'),
                'album': song_details.get('albumName'),
                'duration': song_details.get('duration'),
                'synced_lyrics': song_details.get('syncedLyrics'),
                'plain_lyrics': song_details.get('plainLyrics')
            }
        else:
            print("No synced lyrics available for this song")
            return {
                'id': song_details.get('id'),
                'name': song_details.get('name'),
                'artist': song_details.get('artistName'),
                'album': song_details.get('albumName'),
                'duration': song_details.get('duration'),
                'synced_lyrics': None,
                'plain_lyrics': song_details.get('plainLyrics')
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def prepend_lyrics(lyrics_data: Dict[str, Any])->Dict[str, Any]:
    prepended_lyrics = "[00:00.00]\n"
    
    synced = lyrics_data.get("synced_lyrics")

    if synced is not None:
        prepended_lyrics += synced
    else:
        print("Prepending failed. Synced lyrics are not found.")
    
    lyrics_data["synced_lyrics"] = prepended_lyrics
    return lyrics_data

def get_synced_lyrics(lyrics_data: Dict[str, Any]):
    if not lyrics_data:
        print("No lyrics to display")
        return
    
    synced_lyrics = lyrics_data.get('synced_lyrics')

    if synced_lyrics is None:
        return None
    else:
        return synced_lyrics
    
def get_unsynced_lyrics(lyrics_data: Dict[str, Any]):
    if not lyrics_data:
        print("No lyrics to display")
        return
    
    unsynced_lyrics = lyrics_data.get('plain_lyrics')

    if unsynced_lyrics is None:
        return None
    else:
        return unsynced_lyrics
    
def display_lyrics(lyrics_data: Dict[str, Any]) -> None:
    """
    Display the synced lyrics in a readable format
    """
    if not lyrics_data:
        print("No lyrics data to display")
        return
    
    print(f"\nSong: {lyrics_data.get('name', 'Unknown')}")
    print(f"Artist: {lyrics_data.get('artist', 'Unknown')}")
    print(f"Album: {lyrics_data.get('album', 'Unknown')}")
    
    lyrics_data = prepend_lyrics(lyrics_data)
    synced_lyrics = lyrics_data.get('synced_lyrics')
    if synced_lyrics:
        print("\nLyrics:")
        lines = synced_lyrics.split('\n')
        for line in lines:
            if line.strip():
                print(line.strip())
        print(f"({len(synced_lyrics.split(chr(10)))} total lines)")
    else:
        print("No synced lyrics available")

# Example usage
if __name__ == "__main__":
    # Example search
    song_name = "Bohemian Rhapsody"
    artist_name = "Queen"
    
    result = get_lyrics(song_name, artist_name)
    
    if result:
        display_lyrics(result)
        
        # You can access the actual synced lyrics like this:
        # synced_lyrics = result.get('synced_lyrics')
        # Note: Be mindful of copyright when using lyrics content
    else:
        print("Failed to retrieve lyrics")