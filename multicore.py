from transformers import pipeline
import yt_dlp
import os
import warnings
from pydub import AudioSegment
from tqdm import tqdm
import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

# Check the number of CPU cores available
num_cores = multiprocessing.cpu_count()

# Set this variable to True to enable debug mode
debug = True

# Initialize the ASR (Automatic Speech Recognition) pipeline with the Whisper model
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=-1)

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

def transcribe_segment(segment_path):
    return asr_pipeline(segment_path)["text"]

# Create a debug file for writing debug output
if debug:
    debug_file = open(f"{audio_output_directory}/debug_data_{current_datetime}.txt", "w")

# Initialize an empty list to store the future objects
futures = []

# Create a progress bar
progress_bar = tqdm(total=num_segments, position=0, desc="Progress", unit="segment")

# Perform audio-to-text transcription for each segment in parallel
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    for i in range(num_segments):
        start_time = i * segment_duration * 1000
        end_time = min((i + 1) * segment_duration * 1000, len(audio))
        segment = audio[start_time:end_time]
        segment.export(f'{audio_output_directory}/segment_{i + 1}.wav', format="wav")

        segment_path = f'{audio_output_directory}/segment_{i + 1}.wav'
        future = executor.submit(transcribe_segment, segment_path)
        futures.append(future)

# Concatenate the transcription text from each segment
final_transcription = " ".join(future.result() for future in futures)

# Close the debug file
if debug:
    debug_file.close()

# Create the transcript file for the final transcription
transcript_filename = f"{audio_output_directory}/transcript_{video_title}_{current_datetime}.txt"
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
