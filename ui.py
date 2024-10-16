import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd

from predict import preprocess_data, beat_to_seconds, extract_note_features, extract_chain_features, extract_wall_features, extract_bomb_features, load_beatmap, predict_star_rating

class StarRatingPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BSRank")
        root.iconbitmap('icon.ico')

        # Variables to hold file paths and BPM
        self.model_file = None
        self.beatmap_file = None
        self.bpm = tk.StringVar()

        # Create GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Load Model Button
        self.load_model_button = ttk.Button(self.root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(padx=15, pady=5)

        # Load Beatmap Button
        self.load_beatmap_button = ttk.Button(self.root, text="Load Beatmap", command=self.load_beatmap)
        self.load_beatmap_button.pack(padx=15, pady=5)

        # BPM Input
        self.bpm_label = ttk.Label(self.root, text="BPM:")
        self.bpm_label.pack(padx=15, pady=5)

        self.bpm_entry = ttk.Entry(self.root, textvariable=self.bpm)
        self.bpm_entry.pack(padx=15, pady=5)

        # Predict Button
        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(padx=15, pady=10)

        # Output Label
        self.output_label = ttk.Label(self.root, text="", font=("Fira Code", 12))
        self.output_label.pack(padx=15, pady=5)

    def load_model(self):
        # Open file dialog to select the model file
        self.model_file = filedialog.askopenfilename(
            title="Select Model File", filetypes=(("Pickle Files", "*.pkl"), ("All Files", "*.*")))
        if self.model_file:
            messagebox.showinfo("Model Loaded")

    def load_beatmap(self):
        # Open file dialog to select the beatmap file
        self.beatmap_file = filedialog.askopenfilename(
            title="Select Beatmap File", filetypes=(("Beatmap Files", "*.dat"), ("All Files", "*.*")))
        if self.beatmap_file:
            messagebox.showinfo("Beatmap Loaded")

    def predict(self):
        # Check if model and beatmap are loaded
        if not self.model_file or not self.beatmap_file:
            messagebox.showerror("Error", "Please load both the model and beatmap files.")
            return

        # Get BPM input from user
        try:
            bpm_value = float(self.bpm.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid BPM.")
            return

        # Perform prediction using the loaded model and beatmap
        try:
            predicted_value = self.predict_star_rating(self.beatmap_file, self.model_file, bpm_value)
            self.output_label.config(text=f"Prediction: {predicted_value:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def predict_star_rating(self, beatmap_file, model_filename, bpm):
        # Load the model
        model = joblib.load(model_filename)

        # Preprocess and load beatmap features
        X = load_beatmap(beatmap_file, bpm)

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
    # Initialize themed window
    root = ThemedTk(theme="windefault")
    
    # Create app object
    app = StarRatingPredictorGUI(root)

    # Start the GUI event loop
    root.mainloop()
