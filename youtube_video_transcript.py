from transformers import pipeline
import yt_dlp
import os
import warnings
from pydub import AudioSegment
from tqdm import tqdm
import torch
import datetime
import threading
import argparse

parser = argparse.ArgumentParser()
# Add a command-line argument for the string
parser.add_argument('-g', '--gpu', help='Use GPU if available')
parser.add_argument('-d', '--debug', help='Debug mode')
parser.add_argument('-u', '--url', type=str, help='Input url of the youtube video')
parser.add_argument('-m', '--multithread', help='Multithread mode')
parser.add_argument('-t', '--max_threads', type=int, help='Max number of threads to use')


args = parser.parse_args()


""" # Check if a GPU is available and set the device accordingly
GPU = False  # Use GPU if available
debug = False
multithread = False
max_threads = 1
url = "" """


DEFAULT_GPU = False
DEFAULT_DEBUG = False
DEFAULT_MULTITHREAD = False
DEFAULT_MAX_THREADS = 1
DEFAULT_URL = ""


if args.gpu:
    GPU = True
else:
    GPU = DEFAULT_GPU

if args.debug:
    debug = True
else:
    debug = DEFAULT_DEBUG

if args.multithread:
    multithread = True
else:
    multithread = DEFAULT_MULTITHREAD

if args.max_threads:
    max_threads = args.max_threads
else:
    max_threads = DEFAULT_MAX_THREADS

if args.url:
    url = args.url
else:
    url = DEFAULT_URL



if url == "":
    url = input("Enter the YouTube video URL: ")



# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

if not GPU:
    device = -1
    print("GPU disabled")
else:
    if torch.cuda.is_available():
        device = 0  # Use GPU device 0
        print("Using GPU for ASR.")
        multithread = False
    else:
        device = -1  # Use CPU
        print("GPU not available. Using CPU for ASR.")

if multithread and max_threads == DEFAULT_MAX_THREADS:
    # Get the max number of threads 
    max_threads = input("Enter the max number of threads to use: ")
    try:
        max_threads = int(max_threads)
    except ValueError:
        print("Invalid input. Using 1 thread.")
        max_threads = 1

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

# Create a directory to store the log files
log_output_directory = f'{video_title}_{current_datetime}_logs'
os.makedirs(log_output_directory, exist_ok=True)

# Split the audio into segments
audio = AudioSegment.from_wav(audio_file_path)
num_segments = len(audio) // (segment_duration * 1000) + 1

for i in range(num_segments):
    start_time = i * segment_duration * 1000
    end_time = min((i + 1) * segment_duration * 1000, len(audio))
    segment = audio[start_time:end_time]
    segment.export(f'{audio_output_directory}/segment_{i + 1}.wav', format="wav")

# Create a debug file for writing debug output
if debug:
    debug_file = open(f"{log_output_directory}/debug_data_{current_datetime}.txt", "w")

# Initialize an empty string to store the final transcription
final_transcription = ""
final_transcription_obj = ["" for _ in range(num_segments)]

# Create a progress bar
progress_bar = tqdm(total=num_segments, position=0, desc="Progress", unit="segment")


def transcribe(i):
    segment_path = f'{audio_output_directory}/segment_{i + 1}.wav'
    transcription = asr_pipeline(segment_path)
    if debug:
        debug_file.write(f"Transcription for segment {i + 1}: {transcription['text']}\n")
    progress_bar.update(1)
    final_transcription_obj[i] = transcription['text']

# Perform audio-to-text transcription for each segment
if not multithread:
    for i in range(num_segments):
        r = transcribe(i)
else:   
    args = [i for i in range(num_segments)]
    # Divide the args into chunks in base of the max number of threads
    chunks = [args[i:i + max_threads] for i in range(0, len(args), max_threads)]

    for chunk in chunks:
        threads = []
        for arg in chunk:
            t = threading.Thread(target=transcribe, args=(arg,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

progress_bar.close()

# Build the final transcript
for i in range(num_segments):
    final_transcription += f"{final_transcription_obj[i]} "

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
