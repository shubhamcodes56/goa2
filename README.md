# Speculative RAG — Voice Engine

A low-latency, real-time voice-enabled **Speculative Retrieval-Augmented Generation (RAG)** system powered by **FastAPI**, **ElevenLabs Realtime STT/TTS**, **Sentence Transformers**, and **Qdrant Vector Database**.

---

## ⚡ Architecture & Key Features

### 1. Speculative RAG Pipeline
- **Continuous Speech Streaming:** Captures microphone audio using the Web Audio API (PCM 16-bit / 16kHz) and streams chunks in real time over WebSockets.
- **Real-time ElevenLabs STT:** Sub-second speech recognition with partial transcript streaming.
- **Proactive Speculative Retrieval:** As partial transcripts arrive, the system speculatively issues vector searches against Qdrant to pre-warm and rank candidate context documents before the user finishes speaking.
- **Ring Buffer Context Management:** Employs a sliding-window ring buffer (`ContextRingBuffer`) to maintain short-term conversational context across utterances.
- **Guardrail & Safety Filtering:** Validates incoming queries through a security filter (`GUARDRAIL_BLOCKED_PATTERNS`) to block malicious requests before hitting the LLM/retrieval engine.

### 2. High-Performance Frontend UI
- **Modern Dark Interface:** Glassmorphism, animated gradients, and responsive layout.
- **Real-Time Audio Waveform:** Live canvas visualizer tracking speech dynamics.
- **Speculative Pipeline Visualizer:** Stage-by-stage timing breakdown (STT Latency, Speculative Search, Guardrails, Generation, TTS).
- **Audio Replay & TTS Engine:** On-demand speech synthesis replay with ElevenLabs TTS.
- **Chat History & Latency Diagnostics:** Interactive session sidebar, persistent conversation logs, and live latency pings.

---

## 📁 Repository Structure

```
├── server.py              # Unified FastAPI + WebSocket server
├── elevenlabs_stt.py      # ElevenLabs STT/TTS client, Ring Buffer, Speculative Engine
├── chunker.py             # Sentence-level chunking, embedding generation & Qdrant indexing
├── index.html             # Voice & Text Speculative RAG Web UI
├── requirements.txt       # Python package dependencies
├── .env.example           # Environment variable template
├── benchmark_api.py       # Performance & latency benchmarking suite
├── test_search.py         # Vector similarity search test script
├── test_parquet.py        # Dataset loading & verification script
└── get_dataset_info.py    # Dataset inspection script
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- (Optional) ElevenLabs API Key for voice STT/TTS

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/SVSLEGACY/goa-2.git
cd goa-2

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 4. Build the Vector Index (Optional)
To index the multilingual MS-MARCO dataset with Qdrant:
```bash
python chunker.py
```

### 5. Launch the Server
```bash
python server.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 WebSocket Protocol

| Message Type | Direction | Payload Description |
| :--- | :--- | :--- |
| `audio_chunk` | Client ➔ Server | Base64-encoded PCM audio data from microphone |
| `audio_end` | Client ➔ Server | Signals end of speech / microphone mute |
| `text_query` | Client ➔ Server | Text-based query submission |
| `partial_transcript` | Server ➔ Client | Streaming intermediate speech transcript |
| `final_transcript` | Server ➔ Client | Finalized speech transcript |
| `stage_timing` | Server ➔ Client | Latency breakdown across pipeline stages |
| `answer_chunk` | Server ➔ Client | Streaming answer tokens |
| `guardrail_reject` | Server ➔ Client | Notification if query violates safety filters |
| `metrics` | Server ➔ Client | Detailed timing benchmarks and speculative hits |

---

## 🧪 Testing & Benchmarks

Run the vector search test:
```bash
python test_search.py
```

Run latency and API benchmarking:
```bash
python benchmark_api.py
```

---

## 📄 License
This project is distributed under the terms of the project masterplan and repository guidelines.
