import os
import sys
import subprocess
import threading
import json
import time
import re
from datetime import datetime
import argparse
import tkinter as tk
from tkinter import filedialog
import shutil


class DeepLearningManagerEnhanced:
    """Simplified version without UI dependencies"""
    def __init__(self, data_file="video_generator_data.json"):
        self.data_file = data_file
        self.learning_data = self.load_data()
        self.dl_available = False
        self.dl_enabled = False
        self.improvement_level = 1

    def load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            print(f"Error loading data: {e}")
        return {
            "processed_files": [],
            "model_performance": {},
            "deep_learning_enabled": False,
            "improvement_level": 1,
            "total_videos_generated": 0,
            "user_preferences": {},
            "learning_patterns": [],
            "fusion_mode_enabled": True
        }

    def save_data(self):
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
            print("Data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")

    def get_deep_learning_params(self):
        level = self.learning_data.get("improvement_level", 1)
        return {
            "enhancement_level": level,
            "quality_boost": min(level * 0.2, 1.0),
            "processing_optimization": True,
            "advanced_features": level >= 3,
            "auto_enhancement": level >= 4,
            "smart_rendering": level >= 5,
            "dl_available": self.dl_available,
            "model_trained": self.dl_enabled
        }

    def add_processed_file(self, file_path, models_used, video_paths, processing_time):
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "models_used": models_used,
            "video_paths": video_paths,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "file_type": os.path.splitext(file_path)[1]
        }
        self.learning_data["processed_files"].append(file_info)
        self.learning_data["total_videos_generated"] += len(video_paths)
        self.update_improvement_level()
        self.save_data()

    def update_improvement_level(self):
        total_files = len(self.learning_data["processed_files"])
        if total_files >= 50:
            self.learning_data["improvement_level"] = 5
        elif total_files >= 25:
            self.learning_data["improvement_level"] = 4
        elif total_files >= 10:
            self.learning_data["improvement_level"] = 3
        elif total_files >= 5:
            self.learning_data["improvement_level"] = 2
        else:
            self.learning_data["improvement_level"] = 1

class VideoGeneratorCLI:
    def __init__(self, input_file, models=None, download_dir=None):
        self.dl_manager = DeepLearningManagerEnhanced()
        self.input_file_path = input_file
        if not models:
            self.selected_models = ["microsoft/phi-2"]
        else:
            self.selected_models = models
        self.download_dir = download_dir
        self.process_running = False
        self.output_video_paths = []
        self.pending_models = []
        self.total_models = 0
        self.processed_models = 0
        self.fusion_info = {"enabled": False}

    def get_fusion_strategy(self, models):
        if len(models) <= 1:
            return {"strategy": "single", "primary_model": models[0] if models else "microsoft/phi-2"}
        return {
            "strategy": "optimized_primary",
            "primary_model": models[0],
            "secondary_models": models[1:],
            "fusion_type": "intelligent_selection"
        }

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def generate_videos(self):
        if not os.path.isfile(self.input_file_path):
            self.log(f"Error: Input file not found: {self.input_file_path}")
            return

        if not self.selected_models:
            self.log("Error: No models selected")
            return

        if self.process_running:
            self.log("Generation already in progress")
            return

        self.process_running = True
        self.output_video_paths = []
        self.log(f"Starting generation for {len(self.selected_models)} model(s)...")

        fusion_enabled = self.dl_manager.learning_data.get("fusion_mode_enabled", True)

        if fusion_enabled and len(self.selected_models) > 1:
            self.log(f"Fusion enabled: {len(self.selected_models)} models → 1 video")
            fusion_strategy = self.get_fusion_strategy(self.selected_models)
            optimal_model = fusion_strategy["primary_model"]
            self.log(f"Optimal model: {optimal_model}")
            self.pending_models = [optimal_model]
            self.total_models = 1
            self.fusion_info = {
                "enabled": True,
                "original_models": self.selected_models,
                "strategy": fusion_strategy
            }
        else:
            if len(self.selected_models) > 1:
                self.log(f"Separate mode: {len(self.selected_models)} videos")
            self.pending_models = self.selected_models
            self.total_models = len(self.selected_models)
            self.fusion_info = {"enabled": False}

        self.processed_models = 0
        self.process_next_model()

    def process_next_model(self):
        if not self.pending_models:
            self.log("All models processed.")
            self.process_running = False
            self.dl_manager.add_processed_file(
                self.input_file_path, self.selected_models, self.output_video_paths, 0
            )
            if self.download_dir:
                self.download_videos()
            return

        self.model_name = self.pending_models.pop(0)
        self.processed_models += 1
        self.log(f"Processing model {self.processed_models}/{self.total_models}: {self.model_name}")
        self.run_generation_process(self.model_name)

    def run_generation_process(self, model_name):
        input_filename = os.path.basename(self.input_file_path)
        input_name, _ = os.path.splitext(input_filename)
        model_safe_name = model_name.replace('/', '-')

        if self.fusion_info["enabled"]:
            original_models = self.fusion_info["original_models"]
            models_short = [model.split('/')[-1] for model in original_models[:2]]
            fusion_name = "+".join(models_short)
            if self.dl_manager.learning_data.get("deep_learning_enabled", False):
                level = self.dl_manager.learning_data["improvement_level"]
                dl_params = self.dl_manager.get_deep_learning_params()
                dl_suffix = "TF" if dl_params.get("dl_available", False) else "H"
                output_filename = f"{input_name}_FUSION_{fusion_name}_{dl_suffix}_L{level}.mp4"
            else:
                output_filename = f"{input_name}_FUSION_{fusion_name}.mp4"
        else:
            if self.dl_manager.learning_data.get("deep_learning_enabled", False):
                level = self.dl_manager.learning_data["improvement_level"]
                dl_params = self.dl_manager.get_deep_learning_params()
                dl_suffix = "TF" if dl_params.get("dl_available", False) else "H"
                output_filename = f"{input_name}_{model_safe_name}_{dl_suffix}_L{level}.mp4"
            else:
                output_filename = f"{input_name}_{model_safe_name}.mp4"

        command = [
            sys.executable, "video.py", self.input_file_path,
            "--model", model_name
        ]

        self.log(f"Running model: {model_name}")
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            generated_video_path = None
            for line in iter(process.stdout.readline, ""):
                print(line.strip())
                if ".mp4" in line:
                    import re
                    match = re.search(r': (.+?\.mp4)', line)
                    if match:
                        generated_video_path = match.group(1)
            process.wait()

            if process.returncode == 0 and generated_video_path and os.path.exists(generated_video_path):
                new_path = os.path.join(os.path.dirname(generated_video_path), output_filename)
                if generated_video_path != new_path:
                    os.rename(generated_video_path, new_path)
                    generated_video_path = new_path
                self.log(f"Video generated: {generated_video_path}")
                self.output_video_paths.append(generated_video_path)
            else:
                self.log(f"Error: Generation failed for model {model_name}")

        except Exception as e:
            self.log(f"Exception during generation: {e}")

        self.process_next_model()

    def download_videos(self):
        if not self.output_video_paths:
            self.log("No videos to download")
            return
        if not os.path.isdir(self.download_dir):
            self.log(f"Invalid download directory: {self.download_dir}")
            return
        for video_path in self.output_video_paths:
            if os.path.exists(video_path):
                dest_path = os.path.join(self.download_dir, os.path.basename(video_path))
                try:
                    shutil.copy2(video_path, dest_path)
                    self.log(f"Copied {video_path} to {dest_path}")
                except Exception as e:
                    self.log(f"Error copying {video_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI Video Generator CLI")
    parser.add_argument("input_file", nargs='?', help="Input document file path")
    parser.add_argument("--models", nargs="+", default=None, help="List of AI models to use")
    parser.add_argument("--download_dir", help="Directory to copy generated videos to")
    args = parser.parse_args()

#    if args.input_file:
#        input_file = args.input_file
#    else:       
    root = tk.Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(title="Select input file")
    if not input_file:
        print("No input file selected, exiting.")
        return

    generator = VideoGeneratorCLI(input_file, args.models, args.download_dir)
    generator.generate_videos()

if __name__ == "__main__":
    main()
