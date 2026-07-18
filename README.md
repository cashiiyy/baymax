# BAYMAX AI – Backend System

A production-quality multimodal AI healthcare assistant inspired by Disney's BAYMAX. This backend architecture combines speech recognition, facial emotion detection, medical knowledge retrieval (RAG), LLM response generation, and voice synthesis into a single, cohesive AI pipeline.

## Features

- **Audio Capture & STT**: Voice Activity Detection (VAD) coupled with Faster Whisper for low-latency speech transcription.
- **Emotion Recognition**: MediaPipe face detection + DeepFace to analyze user emotions via webcam, allowing the LLM to provide empathetic responses.
- **Medical RAG**: ChromaDB vector store containing datasets for Diseases, Symptoms, Medicines, and First Aid, ensuring the LLM is always medically grounded.
- **Memory Engine**: SQLite for short-term history and ChromaDB for long-term episodic memory, allowing BAYMAX to remember previous user facts.
- **LLM Reasoning**: Powered by `Qwen3-8B-Instruct`, configured with 8-bit quantization for deployment on consumer GPUs (e.g., RTX 5050 8GB).
- **Voice Synthesis**: Coqui XTTS v2 for natural-sounding voice generation and voice cloning.
- **Avatar WebSocket**: Real-time broadcast of lip-sync (visemes) and emotion commands for a 3D frontend (Unity/Blender).

## Prerequisites

- **OS**: Windows 11 / Linux
- **Python**: 3.11
- **GPU**: NVIDIA GPU with at least 8GB VRAM (CUDA 12.1+ recommended)

## Installation

1. **Clone and Setup Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Add Custom Datasets (Optional but recommended):**
   Place your custom medical CSV or JSON files into `app/datasets/raw/`.
   - Naming convention: `disease*.csv`, `symptom*.csv`, `medicine*.csv`, `firstaid*.csv`.

3. **Provide BAYMAX Voice Reference:**
   Place a 5-10 second clear audio clip of BAYMAX's voice at `models/baymax_voice_ref.wav` for XTTS voice cloning. If omitted, a default voice is used.

4. **Configuration:**
   Review `config.py` for default settings. You can override any setting by creating a `.env` file in the root directory.

## First Run (Dataset Ingestion)

Before running the API, you should build the RAG database from your raw datasets.
You can trigger this programmatically or via the API:
```python
from app.rag.pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.build()
```

## Running the Server

Start the FastAPI backend:
```powershell
python main.py
```

The server will start on `http://localhost:8000`.

- **Swagger UI**: `http://localhost:8000/docs`
- **Avatar WebSocket**: `ws://localhost:8000/avatar/stream`

## Pipeline Architecture

1. **User Speaks / Looks at Camera**
2. `stt/whisper_engine.py` transcribes audio.
3. `emotion/deepface_engine.py` detects emotion (e.g., "sad", "fear").
4. `rag/pipeline.py` fetches relevant medical documents based on transcript.
5. `memory/memory_manager.py` fetches recent conversation history + long-term episodic facts.
6. `context/builder.py` fuses all signals into a strict system prompt.
7. `llm/qwen_engine.py` generates the text response.
8. `llm/response_parser.py` extracts safety flags and avatar expression cues.
9. `tts/xtts_engine.py` synthesizes speech audio.
10. `avatar/controller.py` maps text to phonemes and broadcasts audio/commands over WebSocket.

## Note on vLLM
This repository uses HuggingFace `transformers` + `bitsandbytes` (8-bit) natively for Windows compatibility. If you migrate this stack to a Linux environment, swap the QwenEngine backend for `vLLM` to achieve massive throughput improvements.
