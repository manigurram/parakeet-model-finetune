# ASR Fine-Tuning with Parakeet-TDT Model

This repository contains code to fine-tune the `parakeet-tdt-0.6b-v2` ASR model using NVIDIA NeMo. The fine-tuning process uses `.json` manifest files with audio-text pairs and runs on GPU using PyTorch Lightning.

---

## System Requirements
* NVIDIA GPU with CUDA support (recommended)
* Python 3.10 or higher

```bash
pip install -r requirements.txt
```
## 🔗 Links
* Model : [`parakeet-tdt-0.6b-v2`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt-0.6b-v2)
* Dataset: [`SPRINGLab/asr-task-data`](https://huggingface.co/datasets/SPRINGLab/asr-task-data)

## Data Manifest Format
Each line in the manifest file should be a JSON objec:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "duration": 5.23,
  "text": "hello world"
}
```

* `audio_filepath`: Path to the audio file (WAV format)
* `duration`: Duration of the audio in seconds
* `text`: Corresponding transcription

## ✅ Usage
```bash
python app.py
```

