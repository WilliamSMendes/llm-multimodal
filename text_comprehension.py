import os
import time
import wave
import torch
import langid
import openai
import pyaudio
import whisper
import argparse
import se_extractor
from zipfile import ZipFile
from openai import OpenAI
import speech_recognition as sr
from api import BaseSpeakerTTS, ToneColorConverter


# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Define the name of the log file
chat_log_filename = r"text/logs/chatbot_conversation_log.txt"

# Model and device setup
en_ckpt_base = "checkpoints/base_speakers/EN"
ckpt_converter = "checkpoints/converter"
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = r"text/output"
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)


def open_file(filepath):
    """
    This function opens a file in read mode with UTF-8 encoding and returns its content.

    The function uses a context manager to ensure that the file is properly closed after it is no longer needed.

    Parameters:
    filepath (str): The path to the file to open. This should be a valid path to a file.

    Returns:
    str: The content of the file as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def play_audio(file_path):
    """
    This function plays an audio file using the wave and pyaudio libraries.

    The function first opens the audio file in read-binary mode using the wave library. It then creates a PyAudio instance.

    The function then opens a stream using the PyAudio instance. The stream's format, channels, rate, and output are set based on the 
    properties of the audio file.

    The function then enters a loop where it reads frames from the audio file and writes them to the stream. This continues until there 
    are no more frames to read from the audio file.

    Finally, the function stops the stream, closes it, and terminates the PyAudio instance.

    Parameters:
    file_path (str): The path to the audio file to play. This should be a valid path to a file in a format that the wave library can handle.

    Returns:
    None
    """
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
    """
    This function processes a text prompt and generates an audio file from it using a text-to-speech model and a tone color converter.

    The function first initializes the text-to-speech model and the source style embedding based on the specified style.

    It then extracts the style embedding from the speaker's audio file and generates an audio file from the text prompt using the text-to-speech model.

    The function then uses the tone color converter to convert the style of the generated audio to match the style of the speaker's audio.

    If the audio is generated successfully, it is played. If an error occurs during the process, it is caught and printed.

    Parameters:
    prompt (str): The text prompt to process. This should be a string that forms a valid prompt for the text-to-speech model.
    style (str): The style to use for the text-to-speech model. This should be a string that is recognized by the model.
    audio_file_pth (str): The path to the speaker's audio file. This should be a valid path to a file in a format that the style embedding extractor can handle.

    Returns:
    None
    """
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

        print("Audio generated successfully.")
        play_audio(src_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")


def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    """
    This function sends a query to a local model, streams the response, and prints each full line in neon green color.
    It also logs the conversation to a file.

    The function first constructs a list of messages that includes a system message, the conversation history, and the user's input.

    It then sends a request to the local model to create a chat completion. The response is streamed, meaning that it is received in chunks.

    The function then enters a loop where it processes each chunk of the response. If the chunk contains content, it is added to a line buffer.

    If the line buffer contains a newline character, it is split into lines. Each line is printed, added to the full response, and logged to a file.

    If the line buffer contains a line that does not end with a newline character, it is printed, added to the full response, and logged to a file.

    Finally, the function returns the full response.

    Parameters:
    user_input (str): The user's input to send to the model.
    system_message (str): A system message to include in the messages sent to the model.
    conversation_history (list): The conversation history to include in the messages sent to the model.
    bot_name (str): The name of the bot to use when logging the conversation.

    Returns:
    str: The full response from the model.
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    temperature=1
    
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True
    )

    full_response = ""
    line_buffer = ""

    with open(chat_log_filename, "a") as log_file:  # Open the log file in append mode
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content

            if delta_content is not None:
                line_buffer += delta_content

                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(NEON_GREEN + line + RESET_COLOR)
                        full_response += line + '\n'
                        log_file.write(f"{bot_name}: {line}\n")  # Log the line with the bot's name
                    line_buffer = lines[-1]

        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer
            log_file.write(f"{bot_name}: {line_buffer}\n")  # Log the remaining line

    return full_response

def transcribe_with_whisper(audio_file_path):
    """
    This function transcribes an audio file using the Whisper ASR (Automatic Speech Recognition) model.

    The function first loads the Whisper model. The model size is set to 'base.en', but other sizes like 'tiny', 'base', 'small', 'medium', 'large' can also be used.

    The function then transcribes the audio file using the model. The transcription result is a dictionary that includes the transcribed text among other information.

    Finally, the function returns the transcribed text.

    Parameters:
    audio_file_path (str): The path to the audio file to transcribe. This should be a valid path to a file in a format that the Whisper model can handle.

    Returns:
    str: The transcribed text.
    """
    # Load the model
    model = whisper.load_model("base.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'

    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]


def record_audio(file_path):
    """
    This function records audio from the default input device and saves it to a file.

    The function first creates a PyAudio instance and opens a stream for recording. The stream's format, channels, rate, input, and frames per buffer are set.

    The function then enters a loop where it reads data from the stream and appends it to a list of frames. This continues until the user interrupts the process.

    After the recording is stopped, the stream is stopped and closed, and the PyAudio instance is terminated.

    The function then opens the specified file in write-binary mode using the wave library. It sets the file's channels, sample width, and frame rate, and writes the recorded frames to it.

    Finally, the function closes the file.

    Parameters:
    file_path (str): The path to the file to save the recorded audio to. This should be a valid path to a file in a format that the wave library can handle.

    Returns:
    None
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []

    print("Recording...")

    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def user_chatbot_conversation():
    """
    This function facilitates a conversation between the user and a chatbot.

    The function first initializes an empty conversation history and reads a system message from a file.

    It then enters a loop where it records the user's input, transcribes it, and removes the temporary audio file.

    If the user's input is 'exit', the loop is broken and the function ends.

    The user's input is then printed and added to the conversation history. The chatbot's response is generated and added to the conversation history.

    The chatbot's response is then processed and played as audio.

    If the conversation history exceeds 20 messages, the oldest messages are removed to keep the history at 20 messages.

    Parameters:
    None

    Returns:
    None
    """
    conversation_history = []
    system_message = open_file("chatbot1.txt")
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Julie:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        prompt2 = chatbot_response
        style = "default"
        audio_file_pth2 = r"C:\Users\will\Desktop\QnA_image\low-latency-sts-main\resources\example_reference.mp3"
        process_and_play(prompt2, style, audio_file_pth2)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
  

if __name__ == "__main__":
    user_chatbot_conversation()