# Whisper ASR Subtitle Generation Service

[繁體中文](./README.zh-TW.md)

An automatic speech recognition (ASR) service powered by OpenAI Whisper, converting audio files, videos, or YouTube videos into SRT subtitle files.

## Features

- 🎙️ **Multiple Input Methods**: Upload audio/video files or enter YouTube URLs
- 🌍 **Multi-language Support**: Chinese, English, Japanese, and many more
- 🔄 **Dual Modes**: Transcribe (original language) or Translate (to English)
- 🎯 **VAD Speech Detection**: Precise speech segmentation using Silero VAD
- 📝 **SRT Output**: Standard SRT format ready for video subtitles
- 🚀 **GPU Acceleration**: Multi-GPU parallel processing support

## System Requirements

- Ubuntu Server 24.04
- Docker & Docker Compose
- NVIDIA GPU (RTX 2080 Ti or higher recommended)
- NVIDIA Container Toolkit

## Quick Start

### 1. Install Docker

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group (optional, avoids using sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### 2. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Verify installation
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 3. Build and Start Service

```bash
# Clone the repository
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs

# Build Docker image
docker compose build

# Start service
docker compose up -d

# View logs
docker compose logs -f
```

### 4. Access the Service

Open your browser and navigate to: `http://your-server-ip`

## Configuration

### Environment Variables

Configure the following environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Compute device (`cuda` or `cpu`) |
| `WHISPER_COMPUTE_TYPE` | `float16` | Compute precision (`float16`, `int8`, `float32`) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | Available GPU indices |
| `PRELOAD_MODEL` | `false` | Preload model on startup |

### Available Models

| Model | VRAM Required | Speed | Quality |
|-------|---------------|-------|---------|
| `tiny` | ~1 GB | Fastest | Fair |
| `base` | ~1 GB | Very Fast | Fair |
| `small` | ~2 GB | Fast | Good |
| `medium` | ~5 GB | Medium | Very Good |
| `large-v2` | ~10 GB | Slower | Excellent |
| `large-v3` | ~10 GB | Slower | Best |
| `large-v3-turbo` | ~6 GB | Fast | Excellent |

## Usage

### Upload Audio or Video

1. Click the "Upload audio or video" area
2. Select audio (`.wav`, `.mp3`, `.m4a`, `.flac`) or video (`.mp4`, `.mkv`, `.webm`)
3. Configure language and transcription mode
4. Click "Start Transcription"

### Use YouTube URL

1. Paste the video URL in the "YouTube URL" field
2. Supported formats:
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
   - `https://www.youtube.com/shorts/VIDEO_ID`
3. Configure language and transcription mode
4. Click "Start Transcription"

### Settings

- **Model Size**: Larger models have better quality but slower speed
- **Language**: Select "Auto Detect" or specify a language
- **Task**:
  - Transcribe: Output subtitles in the original language
  - Translate: Translate subtitles to English
- **VAD Speech Detection**: Enable for improved segmentation accuracy
- **Merge Short Subtitles**: Combine short subtitles to appropriate length

## API Usage

Gradio provides an auto-generated API that can be called via Python:

```python
from gradio_client import Client

client = Client("http://your-server-ip")

# Transcribe uploaded file
result = client.predict(
    audio_file="/path/to/audio.wav",
    youtube_url="",
    model_size="large-v3",
    language="auto",
    task="transcribe",
    use_vad=True,
    merge_subtitles=True,
    max_chars=80,
    api_name="/process_audio"
)

srt_content, srt_file_path, status = result
print(status)
print(srt_content)
```

## Project Structure

```
whisper-for-subs/
├── app.py                 # Gradio main application
├── transcriber.py         # Whisper transcription logic
├── vad.py                 # Silero VAD processing
├── youtube_downloader.py  # YouTube download utility
├── srt_utils.py           # SRT format utilities
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # Documentation
```

## Troubleshooting

### GPU Not Available

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

- Use a smaller model (e.g., `medium` or `small`)
- Set `WHISPER_COMPUTE_TYPE=int8` to reduce VRAM usage

### YouTube Download Failed

- Check network connection
- Update yt-dlp: `pip install -U yt-dlp`
- Check if the video has regional restrictions

## License

MIT License

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Gradio](https://gradio.app/)
