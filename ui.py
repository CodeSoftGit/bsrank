import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import threading
import os
import json

from predict import (
    preprocess_data,
    beat_to_seconds,
    extract_note_features,
    extract_chain_features,
    extract_wall_features,
    extract_bomb_features,
    load_beatmap,
    predict_star_rating,
)
from train import train_and_save_model


class StarRatingPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BSRank")
        root.iconbitmap("icon.ico")

        # Variables to hold file paths and BPM
        self.model_file = None
        self.beatmap_file = None
        self.dataset_path = None
        self.model_output_path = None
        self.bpm = tk.StringVar()

        # Create Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(padx=10, pady=10, fill="both", expand=True)

        # Create frames for each tab
        self.train_frame = ttk.Frame(self.notebook)
        self.predict_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        # Add frames to notebook
        self.notebook.add(self.train_frame, text="Train")
        self.notebook.add(self.predict_frame, text="Predict")
        self.notebook.add(self.settings_frame, text="Settings")

        # Create widgets for each tab
        self.create_train_widgets()
        self.create_predict_widgets()
        self.create_settings_widgets()

        # Load settings
        self.load_settings()

    def create_train_widgets(self):
        self.train_dataset_button = ttk.Button(
            self.train_frame, text="Select Dataset", command=self.select_dataset
        )
        self.train_dataset_button.pack(padx=15, pady=5)

        self.train_model_output_button = ttk.Button(
            self.train_frame,
            text="Select Output Path",
            command=self.select_model_output,
        )
        self.train_model_output_button.pack(padx=15, pady=5)

        self.train_button = ttk.Button(
            self.train_frame, text="Train Model", command=self.train_model
        )
        self.train_button.pack(padx=15, pady=10)

        self.progress = ttk.Progressbar(
            self.train_frame, orient="horizontal", length=200, mode="determinate"
        )
        self.progress.pack(padx=15, pady=10)

    def create_predict_widgets(self):
        # Prediction tab widgets
        self.load_model_button = ttk.Button(
            self.predict_frame, text="Load Model", command=self.load_model
        )
        self.load_model_button.pack(padx=15, pady=5)

        self.load_beatmap_button = ttk.Button(
            self.predict_frame, text="Load Beatmap", command=self.load_beatmap
        )
        self.load_beatmap_button.pack(padx=15, pady=5)

        self.bpm_label = ttk.Label(self.predict_frame, text="BPM:")
        self.bpm_label.pack(padx=15, pady=5)

        self.bpm_entry = ttk.Entry(self.predict_frame, textvariable=self.bpm)
        self.bpm_entry.pack(padx=15, pady=5)

        self.predict_button = ttk.Button(
            self.predict_frame, text="Predict", command=self.predict
        )
        self.predict_button.pack(padx=15, pady=10)

        self.output_label = ttk.Label(self.predict_frame, text="")
        self.output_label.pack(padx=15, pady=5)

    def create_settings_widgets(self):
        self.theme_label = ttk.Label(self.settings_frame, text="Select Theme:")
        self.theme_label.pack(padx=15, pady=5)

        self.theme_var = tk.StringVar()
        self.theme_combobox = ttk.Combobox(
            self.settings_frame, textvariable=self.theme_var
        )
        self.theme_combobox["values"] = self.root.get_themes()
        self.theme_combobox.pack(padx=15, pady=5)

        self.apply_theme_button = ttk.Button(
            self.settings_frame, text="Apply Theme", command=self.apply_theme
        )
        self.apply_theme_button.pack(padx=15, pady=10)

        self.clear_settings_button = ttk.Button(
            self.settings_frame, text="Clear Settings", command=self.clear_settings
        )
        self.clear_settings_button.pack(padx=15, pady=10)

    def apply_theme(self):
        selected_theme = self.theme_var.get()
        if selected_theme:
            self.root.set_theme(selected_theme)
            self.save_settings()

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Dataset Selected", f"Dataset Selected")

    def select_model_output(self):
        self.model_output_path = filedialog.asksaveasfilename(
            title="Select Model Output File",
            defaultextension=".pkl",
            filetypes=(("Pickle Files", "*.pkl"), ("All Files", "*.*")),
        )
        if self.model_output_path:
            messagebox.showinfo(
                "Output Path Selected", f"Output Path: {self.model_output_path}"
            )

    def load_model(self):
        self.model_file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("Pickle Files", "*.pkl"), ("All Files", "*.*")),
        )
        if self.model_file:
            messagebox.showinfo("Model Loaded")

    def load_beatmap(self):
        self.beatmap_file = filedialog.askopenfilename(
            title="Select Beatmap File",
            filetypes=(("Beatmap Files", "*.dat"), ("All Files", "*.*")),
        )
        if self.beatmap_file:
            messagebox.showinfo("Beatmap Loaded")

    def predict(self):
        if not self.model_file or not self.beatmap_file:
            messagebox.showerror(
                "Error", "Please load both the model and beatmap files."
            )
            return

        try:
            bpm_value = float(self.bpm.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid BPM.")
            return

        try:
            predicted_value = self.predict_star_rating(
                self.beatmap_file, self.model_file, bpm_value
            )
            self.output_label.config(text=f"Prediction: {predicted_value:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def predict_star_rating(self, beatmap_file, model_filename, bpm):
        model = joblib.load(model_filename)
        X = load_beatmap(beatmap_file, bpm)

        missing_columns = [
            col for col in model.feature_names_in_ if col not in X.columns
        ]
        if missing_columns:
            for col in missing_columns:
                X[col] = 0

        extra_columns = [col for col in X.columns if col not in model.feature_names_in_]
        if extra_columns:
            X = X.drop(columns=extra_columns)

        predicted_star_rating = model.predict(X)
        avg_rating = predicted_star_rating.mean()
        return avg_rating

    def train_model(self):
        if not self.dataset_path or not self.model_output_path:
            messagebox.showerror(
                "Error", "Please provide both dataset path and model output path."
            )
            return

        # Check if dataset folder has at least 1 folder and 1 .dat file
        if not any(
            os.path.isdir(os.path.join(self.dataset_path, f))
            for f in os.listdir(self.dataset_path)
        ):
            messagebox.showerror(
                "Error", "Dataset folder must contain at least one subfolder."
            )
            return

        if not any(f.endswith(".dat") for f in os.listdir(self.dataset_path)):
            messagebox.showerror(
                "Error", "Dataset folder must contain at least one .dat file."
            )
            return

        def run_training():
            try:
                train_and_save_model(self.dataset_path, self.model_output_path)
                messagebox.showinfo(
                    "Training Complete", "Model trained and saved successfully."
                )
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                print(e)

        threading.Thread(target=run_training).start()

    def save_settings(self):
        settings = {"theme": self.theme_var.get()}
        appdata_path = os.path.join(os.getenv("APPDATA"), "BSRank")
        os.makedirs(appdata_path, exist_ok=True)
        with open(os.path.join(appdata_path, "settings.json"), "w") as f:
            json.dump(settings, f)

    def load_settings(self):
        appdata_path = os.path.join(os.getenv("APPDATA"), "BSRank")
        settings_file = os.path.join(appdata_path, "settings.json")
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = json.load(f)
                self.theme_var.set(settings.get("theme", ""))
                if self.theme_var.get():
                    self.root.set_theme(self.theme_var.get())

    def clear_settings(self):
        appdata_path = os.path.join(os.getenv("APPDATA"), "BSRank")
        settings_file = os.path.join(appdata_path, "settings.json")
        if os.path.exists(settings_file):
            os.remove(settings_file)
        self.theme_var.set("")
        messagebox.showinfo("Settings Cleared", "Settings have been cleared.")


if __name__ == "__main__":
    root = ThemedTk(theme="breeze")  # default theme goes here
    app = StarRatingPredictorGUI(root)
    root.mainloop()
