import os
import json
import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Preprocess data for notes, chains, walls, and bombs
def preprocess_data(data, filename_bpm):
    bpm_events = sorted(data.get("bpmEvents", []), key=lambda x: x["b"])

    # Add a default BPM event if none exist (based on the filename BPM)
    if not bpm_events:
        bpm_events.append({"b": 0, "m": filename_bpm})

    notes = []
    for note in data["colorNotes"]:
        if note.get("uninteractable", False):
            continue  # Skip uninteractable notes
        custom_data = note.get("customData", {})
        if "coordinates" in custom_data:
            note["x"], note["y"] = [coord + 3 for coord in custom_data["coordinates"]]
        if "localRotation" in custom_data:
            note["a"] = custom_data["localRotation"][2]  # Use z-rotation for angle
        note["time_seconds"] = beat_to_seconds(note["b"], bpm_events)
        notes.append(note)

    chains = data.get("sliders", []) + data.get("burstSliders", [])
    for chain in chains:
        chain["time_seconds"] = beat_to_seconds(chain["b"], bpm_events)

    obstacles = []
    for wall in data.get("obstacles", []):
        if wall.get("uninteractable", False):
            continue  # Skip uninteractable walls
        custom_data = wall.get("customData", {})
        if "coordinates" in custom_data:
            wall["x"], wall["y"] = [coord + 3 for coord in custom_data["coordinates"]]
        if "size" in custom_data:
            wall["w"], wall["h"] = custom_data["size"]  # Replace width and height
        wall["time_seconds"] = beat_to_seconds(wall["b"], bpm_events)
        obstacles.append(wall)

    bombs = []
    for bomb in data.get("bombNotes", []):
        if bomb.get("uninteractable", False):
            continue  # Skip uninteractable bombs
        custom_data = bomb.get("customData", {})
        if "coordinates" in custom_data:
            bomb["x"], bomb["y"] = [coord + 3 for coord in custom_data["coordinates"]]
        bomb["time_seconds"] = beat_to_seconds(bomb["b"], bpm_events)
        bombs.append(bomb)

    return notes, chains, obstacles, bombs


# Convert beats to seconds using BPM events
def beat_to_seconds(beat, bpm_events):
    total_seconds = 0.0
    prev_beat = 0
    for i in range(len(bpm_events)):
        bpm_event = bpm_events[i]
        bpm = bpm_event["m"]
        start_beat = bpm_event["b"]
        if beat < start_beat:
            total_seconds += (beat - prev_beat) * 60.0 / bpm
            break
        else:
            total_seconds += (start_beat - prev_beat) * 60.0 / bpm
            prev_beat = start_beat
    if beat >= prev_beat:
        bpm = bpm_events[-1]["m"]
        total_seconds += (beat - prev_beat) * 60.0 / bpm
    return total_seconds


# Extract note features
def extract_note_features(note):
    return {
        "time_seconds": note["time_seconds"],
        "x": note["x"],
        "y": note["y"],
        "a": note["a"],  # z rotation for notes
        "direction": note["d"],  # Direction (d)
        "is_chain": 0,
        "is_wall": 0,
        "is_bomb": 0,
    }


# Extract chain features
def extract_chain_features(chain):
    return {
        "time_seconds": chain["time_seconds"],
        "x": chain["x"],
        "y": chain["y"],
        "a": 0,  # Chains may not have rotation
        "direction": chain["d"],  # Direction for chains
        "is_chain": 1,
        "is_wall": 0,
        "is_bomb": 0,
    }


# Extract wall features
def extract_wall_features(wall):
    return {
        "time_seconds": wall["time_seconds"],
        "x": wall["x"],
        "y": wall["y"],
        "a": 0,  # Walls don't have rotation like notes
        "direction": 0,  # No direction for walls
        "is_chain": 0,
        "is_wall": 1,  # It's a wall
        "is_bomb": 0,
    }


# Extract bomb features
def extract_bomb_features(bomb):
    return {
        "time_seconds": bomb["time_seconds"],
        "x": bomb["x"],
        "y": bomb["y"],
        "a": 0,  # Bombs may not have rotation
        "direction": 0,  # No direction for bombs
        "is_chain": 0,
        "is_wall": 0,
        "is_bomb": 1,  # It's a bomb
    }


# Load the dataset for training
def load_dataset(dataset_path):
    feature_list = []
    labels = []

    for star_folder in os.listdir(dataset_path):
        star_rating = float(star_folder.replace("star", "").replace("_", "."))
        star_folder_path = os.path.join(dataset_path, star_folder)

        for map_file in os.listdir(star_folder_path):
            map_path = os.path.join(star_folder_path, map_file)

            map_info = re.match(r"(.+)-(\d+)bpm\.dat", map_file)
            if not map_info:
                print(f"Invalid filename format for {map_file}")
                continue

            bpm = int(map_info.group(2))

            with open(map_path, "r") as f:
                beatmap_data = json.load(f)

            # Preprocess the beatmap data
            notes, chains, walls, bombs = preprocess_data(beatmap_data, bpm)

            # Extract features and append to the list
            for note in notes:
                features = extract_note_features(note)
                feature_list.append(features)
                labels.append(star_rating)

            for chain in chains:
                features = extract_chain_features(chain)
                feature_list.append(features)
                labels.append(star_rating)

            for wall in walls:
                features = extract_wall_features(wall)
                feature_list.append(features)
                labels.append(star_rating)

            for bomb in bombs:
                features = extract_bomb_features(bomb)
                feature_list.append(features)
                labels.append(star_rating)

    return pd.DataFrame(feature_list), labels


# Train and save the model
def train_and_save_model(dataset_path, model_output_path):
    # Load the dataset
    X, y = load_dataset(dataset_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test the model and print mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the model to a file
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    dataset_path = "dataset"  # Path to the dataset
    model_output_path = "random_forest_model.pkl"  # Path to save the trained model

    train_and_save_model(dataset_path, model_output_path)
