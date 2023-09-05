from transformers import pipeline
import yt_dlp
import os
import warnings
from pydub import AudioSegment
from tqdm import tqdm
import torch
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Check if a GPU is available and set the device accordingly
GPU = True  # Use GPU if available
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
url = 'https://www.youtube.com/watch?v=pTCxXZh6VyE'  # Replace VIDEO_ID with the actual YouTube video ID

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    video_title = info['title']

# Replace 'your_audio_file.wav' with the path to your audio file
audio_file_path = 'output.wav'

# Define the duration of each segment in seconds (you can adjust this)
segment_duration = 10  # 5 minutes

# Create a directory to store the audio segments
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
audio_output_directory = f'{video_title}_{current_datetime}_segments'
os.makedirs(audio_output_directory, exist_ok=True)

# Split the audio into segments
audio = AudioSegment.from_wav(audio_file_path)
num_segments = len(audio) // (segment_duration * 1000) + 1

def process_segment(segment_number):
    start_time = segment_number * segment_duration * 1000
    end_time = min((segment_number + 1) * segment_duration * 1000, len(audio))
    segment = audio[start_time:end_time]
    segment.export(f'{audio_output_directory}/segment_{segment_number + 1}.wav', format="wav")
    segment_path = f'{audio_output_directory}/segment_{segment_number + 1}.wav'
    transcription = asr_pipeline(segment_path)
    return transcription['text']

# Create a debug file for writing debug output
if debug:
    debug_file = open(f"{audio_output_directory}/debug_data_{current_datetime}.txt", "w")

# Initialize an empty string to store the final transcription
final_transcription = ""

# Create a progress bar
progress_bar = tqdm(total=num_segments, position=0, desc="Progress", unit="segment")

# Use multithreading/multiprocessing only if CPU is used
if device == -1:
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
        futures = [executor.submit(process_segment, i) for i in range(num_segments)]
        for future in tqdm(futures, position=0, desc="Processing", unit="segment"):
            transcription_text = future.result()
            if debug:
                debug_file.write(f"Transcription for segment {i + 1}: {transcription_text}\n")
            progress_bar.update(1)
            final_transcription += transcription_text + " "
else:
    # If GPU is used, process segments sequentially
    for i in range(num_segments):
        transcription = asr_pipeline(f'{audio_output_directory}/segment_{i + 1}.wav')
        if debug:
            debug_file.write(f"Transcription for segment {i + 1}: {transcription['text']}\n")
        progress_bar.update(1)
        final_transcription += transcription["text"] + " "

# Close the debug file
if debug:
    debug_file.close()

# Create the transcript file for the final transcription
transcript_filename = f"{log_output_directory}/transcript_{video_title}_{current_datetime}.txt"
with open(transcript_filename, "w") as transcript_file:
    transcript_file.write("Final Transcription:\n")
    transcript_file.write(final_transcription)

# Print the final transcription if debug is False
if not debug:
    print("Final Transcription:")
    print(final_transcription)

# Delete the audio files
for i in range(num_segments):
    segment_path = f'{audio_output_directory}/segment_{i + 1}.wav'
    os.remove(segment_path)

# Delete the main output file
main_output_path = 'output.wav'
os.remove(main_output_path)

# Delete the main directory for audio segments
os.rmdir(audio_output_directory)

print("Audio files and main directory successfully deleted.")
