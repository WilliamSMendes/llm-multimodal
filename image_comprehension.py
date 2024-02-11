import re
import os
import wave
import torch
import langid
import openai
import whisper
import pyaudio
import warnings
import argparse
import se_extractor
from tqdm import tqdm
from PIL import Image
from queue import Queue
from openai import OpenAI
from zipfile import ZipFile
from threading import Thread
from huggingface_hub import snapshot_download
from api import BaseSpeakerTTS, ToneColorConverter
from moondream import Moondream
import speech_recognition as sr
from transformers import (
    TextIteratorStreamer,
    CodeGenTokenizerFast as Tokenizer,
)
# Suppress warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)

# ANSI escape codes for colors, for styling the terminal output
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Path to the image file
script_dir = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(script_dir, "image/input/f1.png")

# Define the name of the log file
#chat_log_filename = "chatbot_conversation_log.txt"

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'image/output'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def process_image(image_path):
    """
    This function processes an image using a vision encoder and a text model.

    The function first downloads a model snapshot and initializes a VisionEncoder and a TextModel with the downloaded model.

    It then opens the image file at the specified path and displays it. The image is then processed by the vision encoder to obtain
    image embeddings.

    The function finally returns the text model and the image embeddings.

    Parameters:
    image_path (str): The path to the image file to process. This should be a valid path to a file in a format that the Image class can handle.

    Returns:
    TextModel: The text model initialized with the downloaded model.
    numpy.ndarray: The image embeddings obtained by processing the image with the vision encoder.
    """
    model_path = snapshot_download("vikhyatk/moondream1")

    text_model = Tokenizer.from_pretrained(model_path)
    vision_encoder = Moondream.from_pretrained(model_path)
    vision_encoder.eval()

    image = Image.open(image_path)
    image.show()
    print(f"{NEON_GREEN}Processing the image...{RESET_COLOR}")
    image_embeds = vision_encoder.encode_image(image)

    return text_model, image_embeds, vision_encoder


def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()


def process_and_play(prompt, style, audio_file_pth):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se
    speaker_wav = audio_file_pth

    # Process text and generate audio
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)

        #print(f"{NEON_GREEN}Talking...{RESET_COLOR}")
        print(f"{YELLOW}> Assistant: {RESET_COLOR}{prompt}")
        play_audio(src_path)
        print(f"{CYAN}Answered{RESET_COLOR}\n\n")

    except Exception as e:
        print(f"Error during audio generation: {e}")


def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("small.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'

    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]


# Function to record audio from the microphone and save to a file
def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []

    print(f"{NEON_GREEN}Recording...{RESET_COLOR}")

    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print(f"Recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def main():
    # Definitions
    text_model, image_embeds, vision_encoder = process_image(IMG_PATH)
    #chat_history = ""
    audio_file = "temp_recording.wav"
    style = "default"
    audio_file_pth2 = r"voice/sophie.wav"

    while True:
        print(f"{CYAN}Ask a question about the image or say 'exit' to quit{RESET_COLOR}")

        # Record the user's voice input
        record_audio(audio_file)
        print(f"{NEON_GREEN}Processing the audio... {RESET_COLOR}")
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit.":  # Say 'exit' to end the conversation
            break

        print(f"{YELLOW}> You: {RESET_COLOR}{user_input}")
        answer = vision_encoder.answer_question(image_embeds, user_input, text_model)
        #chat_history += f"{CYAN}Question: {RESET_COLOR}{YELLOW}{user_input}{RESET_COLOR}\n\n{CYAN}Answer: {answer}{RESET_COLOR}\n\n"

        # Generate audio response for the assistant's message
        process_and_play(answer, style, audio_file_pth2)
    
if __name__ == "__main__":
    main()