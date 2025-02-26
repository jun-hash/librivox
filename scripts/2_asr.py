#!/usr/bin/env python3
"""
Run Fast Whisper v3 turbo ASR model on audio files from a local folder (GPU only).
"""

import argparse
import os
import time
from pathlib import Path
from math import exp

import torch
import librosa
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel


class ASRProcessor:
    def __init__(self, output_dir: str, force: bool = False):
        """
        Initialize the ASR processor with the Fast Whisper v3 turbo model on GPU.

        Args:
            output_dir: Directory to save ASR results.
            force: Process files even if results already exist.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.processed_count = 0
        self.skipped_count = 0

        start_time = time.time()
        # Force GPU usage
        self.device = "cuda"
        self.compute_type = "float16"
        print("Initializing Fast Whisper model on GPU...")
        self.model = WhisperModel(
            "turbo",
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=8,
            num_workers=7,
        )
        print(f"Model loaded on {self.device}. Initialization time: {time.time() - start_time:.4f} seconds.")

    def process_audio_file(self, audio_path: Path) -> bool:
        """
        Process a single audio file with Fast Whisper ASR.
        """
        file_id = audio_path.stem
        output_file = self.output_dir / f"{file_id}.txt"
        if not self.force and output_file.exists() and output_file.stat().st_size > 0:
            print(f"Skipping {file_id} (already processed).")
            self.skipped_count += 1
            return True

        file_start = time.time()
        print(f"\n=== Processing file: {file_id} ===")

        try:
            audio_load_start = time.time()
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)
            print(f"[1] Audio loading time: {time.time() - audio_load_start:.4f} seconds.")
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return False

        transcribe_start = time.time()
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=4,
            without_timestamps=True
        )
        segments = list(segments)
        print(f"[2] Transcription time: {time.time() - transcribe_start:.4f} seconds.")

        # 초고속 포맷팅
        format_start = time.time()
        
        write_start = time.time()
        # Write each segment with its confidence to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(f"{segment.text.strip()}\n")
                f.write(f"Confidence: {exp(segment.avg_logprob):.4f}\n\n")
        
        print(f"[4] Output writing time: {time.time() - write_start:.4f} seconds.")

        print(f"[5] Total processing time for {file_id}: {time.time() - file_start:.4f} seconds.")
        self.processed_count += 1
        return True

    def process_folder(self, audio_folder: str, extensions=None):
        """
        Process all audio files in the specified folder.
        """
        if extensions is None:
            extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
        folder_path = Path(audio_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Audio folder not found: {folder_path}")

        print("\n=== Starting folder processing ===")
        start_time = time.time()
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder_path.glob(f"*{ext}"))
        print(f"[1] File search time: {time.time() - start_time:.4f} seconds (found {len(audio_files)} files).")

        if not audio_files:
            print(f"No audio files found in {folder_path} with extensions {extensions}.")
            return

        process_start = time.time()
        for audio_file in tqdm(audio_files, desc="Processing files"):
            if not self.process_audio_file(audio_file):
                print(f"Failed to process {audio_file.name}")
        print(f"[2] Total files processing time: {time.time() - process_start:.4f} seconds.")
        print(f"[3] Overall folder processing time: {time.time() - start_time:.4f} seconds.")

        print("\n=== Processing Summary ===")
        print(f"Total files: {len(audio_files)}")
        print(f"Processed: {self.processed_count}")
        print(f"Skipped: {self.skipped_count}")
        if self.processed_count > 0:
            avg_time = (time.time() - process_start) / self.processed_count
            print(f"Average processing time per file: {avg_time:.4f} seconds.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Fast Whisper v3 turbo ASR model on audio files (GPU only)."
    )
    parser.add_argument("--audio-folder", type=str, default="./data/cut", help="Folder containing audio files to process")
    parser.add_argument("--output-dir", type=str, default="./asr_result_text", help="Directory to save ASR results")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"], help="Audio file extensions to process")
    parser.add_argument("--force", action="store_true", help="Force processing of all files, even if already processed")
    args = parser.parse_args()

    overall_start = time.time()
    processor = ASRProcessor(output_dir=args.output_dir, force=args.force)
    processor.process_folder(args.audio_folder, args.extensions)
    print(f"\n=== Total execution time: {time.time() - overall_start:.4f} seconds ===")


if __name__ == "__main__":
    main()
