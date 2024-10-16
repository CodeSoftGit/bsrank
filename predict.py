import json
import re
import joblib
import pandas as pd

# Preprocess data to handle notes, chains, walls, and bombs
def preprocess_data(data, filename_bpm):
    bpm_events = sorted(data.get('bpmEvents', []), key=lambda x: x['b'])

    # Add BPM event if none exist (based on filename BPM)
    if not bpm_events:
        bpm_events.append({'b': 0, 'm': filename_bpm})

    notes = []
    for note in data['colorNotes']:
        if note.get('uninteractable', False):
            continue  # Skip uninteractable notes
        custom_data = note.get('customData', {})
        if 'coordinates' in custom_data:
            note['x'], note['y'] = [coord + 3 for coord in custom_data['coordinates']]
        if 'localRotation' in custom_data:
            note['a'] = custom_data['localRotation'][2]  # Use z-rotation for angle
        note['time_seconds'] = beat_to_seconds(note['b'], bpm_events)
        notes.append(note)

    chains = data.get('sliders', []) + data.get('burstSliders', [])
    for chain in chains:
        chain['time_seconds'] = beat_to_seconds(chain['b'], bpm_events)

    obstacles = []
    for wall in data.get('obstacles', []):
        if wall.get('uninteractable', False):
            continue  # Skip uninteractable walls
        custom_data = wall.get('customData', {})
        if 'coordinates' in custom_data:
            wall['x'], wall['y'] = [coord + 3 for coord in custom_data['coordinates']]
        if 'size' in custom_data:
            wall['w'], wall['h'] = custom_data['size']  # Replace width and height
        wall['time_seconds'] = beat_to_seconds(wall['b'], bpm_events)
        obstacles.append(wall)

    bombs = []
    for bomb in data.get('bombNotes', []):
        if bomb.get('uninteractable', False):
            continue  # Skip uninteractable bombs
        custom_data = bomb.get('customData', {})
        if 'coordinates' in custom_data:
            bomb['x'], bomb['y'] = [coord + 3 for coord in custom_data['coordinates']]
        bomb['time_seconds'] = beat_to_seconds(bomb['b'], bpm_events)
        bombs.append(bomb)

    return notes, chains, obstacles, bombs

# Convert beats to seconds using BPM events
def beat_to_seconds(beat, bpm_events):
    total_seconds = 0.0
    prev_beat = 0
    for i in range(len(bpm_events)):
        bpm_event = bpm_events[i]
        bpm = bpm_event['m']
        start_beat = bpm_event['b']
        if beat < start_beat:
            total_seconds += (beat - prev_beat) * 60.0 / bpm
            break
        else:
            total_seconds += (start_beat - prev_beat) * 60.0 / bpm
            prev_beat = start_beat
    if beat >= prev_beat:
        bpm = bpm_events[-1]['m']
        total_seconds += (beat - prev_beat) * 60.0 / bpm
    return total_seconds

# Extract note features
def extract_note_features(note):
    return {
        'time_seconds': note['time_seconds'],
        'x': note['x'],
        'y': note['y'],
        'a': note['a'],  # z rotation for notes
        'direction': note['d'],  # Direction (d)
        'is_chain': 0,
        'is_wall': 0,
        'is_bomb': 0
    }

# Extract chain features
def extract_chain_features(chain):
    return {
        'time_seconds': chain['time_seconds'],
        'x': chain['x'],
        'y': chain['y'],
        'a': 0,  # Chains may not have rotation
        'direction': chain['d'],  # Direction for chains
        'is_chain': 1,
        'is_wall': 0,
        'is_bomb': 0
    }

# Extract wall features
def extract_wall_features(wall):
    return {
        'time_seconds': wall['time_seconds'],
        'x': wall['x'],
        'y': wall['y'],
        'a': 0,  # Walls don't have rotation like notes
        'direction': 0,  # No direction for walls
        'is_chain': 0,
        'is_wall': 1,  # It's a wall
        'is_bomb': 0
    }

# Extract bomb features
def extract_bomb_features(bomb):
    return {
        'time_seconds': bomb['time_seconds'],
        'x': bomb['x'],
        'y': bomb['y'],
        'a': 0,  # Bombs may not have rotation
        'direction': 0,  # No direction for bombs
        'is_chain': 0,
        'is_wall': 0,
        'is_bomb': 1  # It's a bomb
    }

# Load and preprocess a beatmap for prediction
def load_beatmap(beatmap_path, bpm_from_filename):
    with open(beatmap_path, 'r') as f:
        beatmap_data = json.load(f)

    # Preprocess the beatmap data
    notes, chains, walls, bombs = preprocess_data(beatmap_data, bpm_from_filename)

    # Extract features from notes, chains, walls, and bombs
    features_list = []
    for note in notes:
        features = extract_note_features(note)
        features_list.append(features)

    for chain in chains:
        features = extract_chain_features(chain)
        features_list.append(features)

    for wall in walls:
        features = extract_wall_features(wall)
        features_list.append(features)

    for bomb in bombs:
        features = extract_bomb_features(bomb)
        features_list.append(features)

    return pd.DataFrame(features_list)

# Function to predict the star rating of a beatmap
def predict_star_rating(beatmap_file, model_filename):
    # Load the model
    model = joblib.load(model_filename)

    # Extract BPM from filename (like before)
    map_info = re.match(r'(.+)-(\d+)bpm\.dat', beatmap_file)
    if not map_info:
        print(f"Invalid filename format for {beatmap_file}")
        return

    bpm = int(map_info.group(2))
    beatmap_path = beatmap_file

    # Preprocess and load beatmap features
    X = load_beatmap(beatmap_path, bpm)

    # Ensure the same features as in training
    missing_columns = [col for col in model.feature_names_in_ if col not in X.columns]
    if missing_columns:
        for col in missing_columns:
            X[col] = 0  # Fill missing columns with zeros

    extra_columns = [col for col in X.columns if col not in model.feature_names_in_]
    if extra_columns:
        X = X.drop(columns=extra_columns)  # Drop extra columns

    # Make predictions
    predicted_star_rating = model.predict(X)
    avg_rating = predicted_star_rating.mean()
    return avg_rating

if __name__ == "__main__":
    # Example usage:
    beatmap_file = r'C:\Users\Code\Documents\Programming\bsrank\dataset\13star\amogus-125bpm.dat'
    print(predict_star_rating(beatmap_file, model_filename='random_forest_model.pkl'))
