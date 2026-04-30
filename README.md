# 🎙️ Real-Time Speech-to-Text & Keyword Detection System

A production-grade, real-time audio processing pipeline that transcribes live speech and detects keywords with high accuracy — even in noisy environments with background music.

---

## 🚀 Demo

```
User says: "Hello, I would like to have an ice cream"

Server Response:
{
  "text": "Hello, I would like to have an ice cream",
  "keywords": [
    {
      "keyword": "ice cream",
      "match_word": "ice cream",
      "start": 1.9,
      "end": 2.4,
      "confidence": 1.0,
      "match_type": "exact"
    }
  ],
  "orders": [{ "item": "ice cream", "quantity": 1 }],
  "is_final": false
}
```

---

## 🧠 Architecture

```
Microphone Input (client.py)
        │
        ▼
WebSocket Stream (FastAPI)
        │
        ▼
Audio Buffer (Sliding Window: 5s window, 1.5s hop)
        │
        ▼
┌───────────────────────────────────┐
│         Preprocessing Pipeline    │
│  1. Noise Reduction (noisereduce) │
│  2. Voice Activity Detection      │
│     (Silero VAD)                  │
│  3. Audio Normalization           │
└───────────────────────────────────┘
        │
        ▼
Async Inference Queue (2 workers)
        │
        ▼
faster-whisper Transcription
(medium.en, word-level timestamps)
        │
        ▼
┌───────────────────────────────────┐
│       Keyword Detection Engine    │
│  - Fuzzy Matching (rapidfuzz)     │
│  - Phonetic Matching (jellyfish)  │
│  - Multi-word phrase support      │
│  - Overlap deduplication          │
└───────────────────────────────────┘
        │
        ▼
Entity Extraction (item + quantity)
        │
        ▼
WebSocket Response (JSON)
```

---

## ⚙️ Key Engineering Decisions

### 1. Sliding Window with Context Overlap
Instead of processing audio in isolated chunks, a **5-second sliding window with 1.5s hop** is used. The previous audio tail is prepended as context to each new window — this prevents keywords from being cut off at chunk boundaries.

### 2. Two-Stage VAD
- **Silero VAD** (neural network-based) filters non-speech audio before sending to Whisper
- Minimum speech ratio threshold (20%) prevents wasting inference on silent windows
- Reduces Whisper hallucinations significantly

### 3. Fuzzy + Phonetic Keyword Matching
Standard exact-match fails in noisy environments. This system uses:
- **RapidFuzz** for fuzzy string similarity (threshold varies by keyword length)
- **Jellyfish Soundex** for phonetic matching on borderline cases
- Adaptive thresholds: short keywords (≤3 chars) need 92% similarity; longer ones need 80%

### 4. Async Inference with Semaphore
- Two parallel inference workers using `asyncio`
- `asyncio.Semaphore(2)` prevents CPU overload
- Queue drop policy: oldest item dropped when queue is full (real-time latency > completeness)

### 5. Overlap Deduplication
Sliding windows cause the same text to appear in multiple transcriptions. A custom `remove_overlap()` function compares the tail of the previous emission with the head of the new text to strip duplicates before sending to the client.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Real-time Transport | WebSockets |
| Speech Transcription | faster-whisper (medium.en) |
| Audio Processing | Librosa, SoundFile, NumPy |
| Voice Activity Detection | Silero VAD |
| Noise Reduction | noisereduce |
| Keyword Matching | RapidFuzz, Jellyfish |
| Async Runtime | asyncio, HTTPX |
| Audio Client | sounddevice |

---

## 📁 Project Structure

```
├── app/
│   ├── api/
│   │   └── keyword.py              # WebSocket route
│   ├── config/
│   │   ├── keyword_config.py       # Food keywords list
│   │   └── streaming_config.py     # Audio pipeline constants
│   ├── core/
│   │   └── model.py                # Whisper model + semaphore init
│   └── services/
│       ├── streaming_service.py    # Main WebSocket handler + audio pipeline
│       ├── transcription.py        # Whisper transcription wrapper
│       ├── keyword_detector.py     # Fuzzy + phonetic keyword matching
│       ├── entity_extractor.py     # Item + quantity extraction
│       ├── silero_vad.py           # Neural VAD filtering
│       ├── denoise.py              # Noise reduction
│       ├── remove_overlap.py       # Transcript deduplication
│       └── response_builder.py     # JSON response formatter
├── client.py                       # Microphone streaming test client
└── requirements.txt
```

---

## 🏃 Getting Started

### Prerequisites
- Python 3.9+
- Working microphone

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/realtime-keyword-detection.git
cd realtime-keyword-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run the Client (Microphone Streaming)

```bash
python client.py
```

---

## 🔧 Configuration

Edit `app/config/streaming_config.py` to tune the pipeline:

```python
SAMPLE_RATE = 16000       # Audio sample rate (Hz)
WINDOW_SEC = 5            # Sliding window size
HOP_SEC = 1.5             # Window hop size
CONTEXT_SEC = 2.5         # Context tail prepended to each window
MAX_BUFFER_SEC = 20       # Max audio buffer size
QUEUE_SIZE = 3            # Inference queue depth
```

Edit `app/config/keyword_config.py` to add/remove keywords:

```python
FOOD_KEYWORDS = [
    "pizza", "burger", "fries", "coffee",
    "tea", "juice", "biryani", "dosa",
    "idli", "cake", "ice cream"
]
```

---

## 🧪 Challenges Solved

| Challenge | Solution |
|---|---|
| Background music lyrics detected as speech | Silero VAD filters non-human-speech audio before transcription |
| Same keyword detected multiple times | Timestamp-based deduplication (< 1.0s window) |
| Accented / mispronounced keywords missed | Phonetic matching via Jellyfish Soundex |
| High CPU load from continuous inference | Async semaphore limits concurrent Whisper calls to 2 |
| Text repetition across sliding windows | Custom overlap removal compares tail of previous with head of new |

---

## 📄 License

MIT License — feel free to use and modify.

---

## 👤 Author

**Nitin Negi**
- LinkedIn: [linkedin.com/in/neural-nitin](https://linkedin.com/in/neural-nitin)
- Email: neuralnitin@gmail.com
- Github: https://github.com/Polar6ear
