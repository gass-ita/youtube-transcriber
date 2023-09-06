**Introduction:**

This Python script demonstrates how to perform Automatic Speech Recognition (ASR) on a YouTube video's audio and segment the audio into smaller chunks for transcription using the Hugging Face Transformers library, yt-dlp, and other Python libraries.

**Prerequisites:**

Before using this script, make sure you have the following prerequisites installed:

-   Python 3.x
-   [Hugging Face Transformers](https://huggingface.co/transformers/)
-   [yt-dlp](https://github.com/yt-dlp/yt-dlp)
-   [pydub](https://github.com/jiaaro/pydub)
-   [tqdm](https://github.com/tqdm/tqdm)
-   [torch](https://github.com/pytorch/pytorch)

You can install the required Python packages using pip:

Copy code

`pip install -r requirements.txt` 

**Usage:**

1.  Import the necessary libraries and configure the script to your requirements.
    
2.  Initialize the ASR pipeline using the Whisper model from Hugging Face Transformers. You can specify the GPU device if needed.
    
3.  Set the `ydl_opts` dictionary to configure the video download options. Specify the video URL you want to process.
    
4.  Download the video using yt-dlp and extract the audio into a WAV file.
    
5.  Define the `segment_duration` to specify the duration of each audio segment in seconds.
    
6.  Create an output directory to store the audio segments.
    
7.  Split the downloaded audio into segments based on the defined duration and export them as individual WAV files.
    
8.  Initialize an empty string to store the final transcription.
    
9.  Create a progress bar to track the transcription progress.
    
10.  Iterate through each audio segment, perform ASR on it, and concatenate the transcribed text.
    
11.  Print the final transcription.
    

**Example:**

In the provided script, we download a YouTube video, split its audio into 10-second segments, perform ASR on each segment, and then display the final transcription.

**Customization:**

-   You can adjust the `segment_duration` to control the length of audio segments.
-   Modify the `ydl_opts` dictionary to specify different video download and audio extraction options.
-   Change the ASR model by specifying a different model in the `pipeline` initialization.
-   Customize the output directory and file names for audio segments and the final transcription.
