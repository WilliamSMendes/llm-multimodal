# llm-multimodal

## About

`llm-multimodal` is a sophisticated multi-modal AI tool designed to understand and process various types of inputs including text, speech, images, and videos. The tool aims to provide comprehensive answers based on the content it processes. While it currently supports text, speech, image, and video comprehension, link-based content comprehension is still under development.

## Features

- **Text Comprehension**: Understands and processes textual input using advanced language models.
- **Speech Comprehension**: Transcribes and understands speech inputs using state-of-the-art ASR models.
- **Image Comprehension**: Processes and understands images to provide relevant information and answers.
- **Video Comprehension**: Extracts frames and audio from videos to generate detailed summaries and descriptions.

## Installation

To get started, ensure you have Python installed on your system. Clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/llm-multimodal.git
cd llm-multimodal
pip install -r requirements.txt
```

## Usage

### Image Comprehension

The `image_comprehension.py` script processes images and generates audio responses based on the questions asked about the images.

```sh
python image_comprehension.py
```

### Text Comprehension

The `text_comprehension.py` script facilitates a chatbot-like conversation where the user can interact with the system using text and receive audio responses.

```sh
python text_comprehension.py
```

### Video Comprehension

The `video_comprehension.py` script extracts frames and audio from a video file, processes them to generate a detailed summary.

```sh
python video_comprehension.py
```

## File Descriptions

### `image_comprehension.py`

This script processes images using a vision encoder and text model. It also handles audio recording and playback. The main functionalities include:

- Processing images to generate embeddings.
- Recording audio from the microphone.
- Transcribing speech using the Whisper model.
- Generating audio responses to questions about the images.

### `text_comprehension.py`

This script facilitates text-based interactions using a chatbot model. Key functionalities include:

- Recording and transcribing audio input.
- Interacting with a language model to generate responses.
- Converting text responses to audio.

### `video_comprehension.py`

This script handles video files, extracting frames and audio to generate comprehensive summaries. It includes:

- Extracting frames from a video at specified intervals.
- Converting video audio to text using the Whisper model.
- Generating descriptions for each extracted frame.
- Combining audio and visual descriptions to generate a detailed summary.

### `requirements.txt`

Lists all the dependencies required to run the scripts. Install them using:

```sh
pip install -r requirements.txt
```

## Dependencies

- `librosa`
- `faster-whisper`
- `pydub`
- `wavmark`
- `numpy`
- `eng_to_ipa`
- `inflect`
- `unidecode`
- `whisper-timestamped`
- `openai`
- `python-dotenv`
- `pypinyin`
- `cn2an`
- `jieba`
- `langid`
- `accelerate`
- `huggingface-hub`
- `Pillow`
- `torch`
- `torchvision`
- `transformers`
- `einops`
- `gradio`
- `ffmpeg-python`
- `moviepy`
- `opencv-python`
- `pyaudio`
- `SpeechRecognition`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING](CONTRIBUTING.md) guidelines before starting.

---

Feel free to customize this `README.md` to better fit your project's specific details and requirements.
