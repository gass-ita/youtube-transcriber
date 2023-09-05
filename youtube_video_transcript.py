# -*- coding: utf-8 -*-
"""Youtube Video TranScript

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZjKI9xuW3a6-iyfsHCLRcx7NKiIMsN1T
"""

from transformers import pipeline
import yt_dlp
import os
import warnings
from pydub import AudioSegment
from tqdm import tqdm
import torch
import datetime

# Check if a GPU is available and set the device accordingly


# Set this variable to True to enable debug mode

GPU = True # Use gpu if is avilable
debug = True

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")


if not GPU:
    device = -1
    print("GPU disabled")
else:
    if torch.cuda.is_available():
        device = 0  # Use GPU device 0
        print("Using GPU for ASR.")
    else:
        device = -1  # Use CPU
        print("GPU not available. Using CPU for ASR.")



# Initialize the ASR (Automatic Speech Recognition) pipeline with the Whisper model
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=device)

# Create a yt_dlp instance
ydl_opts = {
    'format': 'bestaudio/best',  # Specify the audio format you want
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',  # Use WAV as the preferred audio codec
    }],
    'outtmpl': 'output',  # Specify the output file name without extension
}
url = 'https://www.youtube.com/watch?v=nB14gW9WdK0'  # Replace VIDEO_ID with the actual YouTube video ID

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    video_title = info['title']

# Replace 'your_audio_file.wav' with the path to your audio file
audio_file_path = 'output.wav'

# Define the duration of each segment in seconds (you can adjust this)
segment_duration = 10  # 5 minutes

# Create a directory to store the segments
output_directory = f'{video_title}_segments'
os.makedirs(output_directory, exist_ok=True)

# Split the audio into segments
audio = AudioSegment.from_wav(audio_file_path)
num_segments = len(audio) // (segment_duration * 1000) + 1

# Get the current date and time for the filenames
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a debug file for writing debug output
if debug:
    debug_file = open(f"debug_data_{current_datetime}.txt", "w")

# Initialize an empty string to store the final transcription
final_transcription = ""

# Create a progress bar
progress_bar = tqdm(total=num_segments, position=0, desc="Progress", unit="segment")

# Perform audio-to-text transcription for each segment
for i in range(num_segments):
    segment_path = f'{output_directory}/segment_{i + 1}.wav'
    transcription = asr_pipeline(segment_path)
    if debug:
        debug_file.write(f"Transcription for segment {i + 1}: {transcription['text']}\n")
    progress_bar.update(1)
    # Concatenate the transcription text from each segment
    final_transcription += transcription["text"] + " "

# Close the debug file
if debug:
    debug_file.close()

# Create the transcript file for the final transcription
transcript_filename = f"transcript_{video_title}_{current_datetime}.txt"
with open(transcript_filename, "w") as transcript_file:
    transcript_file.write("Final Transcription:\n")
    transcript_file.write(final_transcription)

# Print the final transcription if debug is False
print("Final Transcription:")
print(final_transcription)