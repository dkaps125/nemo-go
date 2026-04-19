# nemo-go

Go inference library for [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) ASR models using ONNX Runtime. Supports three model families:

| Family | Example models | Decoding |
|--------|---------------|---------|
| **Parakeet CTC** | `nvidia/parakeet-ctc-0.6b-en` | Greedy CTC |
| **Parakeet TDT / RNNT** | `nvidia/parakeet-tdt-0.6b-v2` | Greedy TDT |
| **Canary** | `nvidia/canary-180m` | Autoregressive (greedy) |

---

## Requirements

- Go 1.24+
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (for ONNX export)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) shared library
- [ffmpeg](https://ffmpeg.org/) on `PATH` (for audio loading)

---

## Installation

### 1. Clone and build

```bash
git clone https://github.com/dkaps125/nemo-go
cd nemo-go
go build ./...
```

### 2. Install ONNX Runtime

Download the ONNX Runtime shared library for your platform from the [official releases](https://github.com/microsoft/onnxruntime/releases).

**macOS (CPU):**
```bash
# Example: onnxruntime-osx-arm64-1.22.0.tgz
tar -xzf onnxruntime-osx-arm64-*.tgz
cp onnxruntime-osx-arm64-*/lib/libonnxruntime.dylib /usr/local/lib/
```

**Linux (CPU):**
```bash
# Example: onnxruntime-linux-x64-1.22.0.tgz
tar -xzf onnxruntime-linux-x64-*.tgz
cp onnxruntime-linux-x64-*/lib/libonnxruntime.so.* /usr/local/lib/
ldconfig
```

Set the path via environment variable (recommended, avoids passing `--ort-lib` every time):
```bash
export ORT_LIB_PATH=/usr/local/lib/libonnxruntime.dylib      # macOS
export ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so.1.22.0  # Linux
```

### 3. Install ffmpeg

```bash
brew install ffmpeg       # macOS
apt-get install ffmpeg    # Debian/Ubuntu
```

---

## Exporting models to ONNX

NeMo models must be exported to ONNX before Go inference. The export script handles downloading, conversion, and metadata generation.

### Setup Python environment

The script uses [uv](https://github.com/astral-sh/uv) for dependency management — no manual `pip install` needed.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Export a model

```bash
# Parakeet CTC (pure CTC)
uv run python3 scripts/export_onnx.py \
    --model parakeet \
    --checkpoint nvidia/parakeet-ctc-0.6b-en \
    --output-dir /tmp/parakeet-ctc-export

# Parakeet TDT (RNNT/TDT — use --model rnnt)
uv run python3 scripts/export_onnx.py \
    --model rnnt \
    --checkpoint nvidia/parakeet-tdt-0.6b-v2 \
    --output-dir /tmp/parakeet-tdt-export

# Canary (multilingual encoder-decoder)
uv run python3 scripts/export_onnx.py \
    --model canary \
    --checkpoint nvidia/canary-180m \
    --output-dir /tmp/canary-export
```

You can also pass a local `.nemo` file instead of a HuggingFace model ID:
```bash
uv run python3 scripts/export_onnx.py \
    --model parakeet \
    --checkpoint /path/to/model.nemo \
    --output-dir /tmp/export
```

### Output files

**Parakeet CTC** (`--model parakeet`):
```
{stem}_encoder_ctc.onnx    — encoder + CTC head
tokenizer.model             — SentencePiece tokenizer
```

**Parakeet TDT / RNNT** (`--model rnnt`):
```
{stem}_encoder.onnx         — encoder
{stem}_decoder_joint.onnx   — LSTM prediction network + joint network (single-step)
{stem}_rnnt_meta.json       — TDT durations, state shapes, mel bin count
tokenizer.model
```

**Canary** (`--model canary`):
```
{stem}_encoder.onnx
{stem}_decoder.onnx
tokenizer.model
```

> **Note:** Large models export with a `.onnx.data` sidecar file containing the weights. Keep it alongside the `.onnx` file.

---

## Running transcription

Build and run the CLI:

```bash
go build -o nemo-transcribe ./cmd/nemo-transcribe
```

### Parakeet CTC

```bash
./nemo-transcribe \
    --model parakeet \
    --checkpoint /tmp/parakeet-ctc-export \
    audio.wav
```

### Parakeet TDT / RNNT

```bash
./nemo-transcribe \
    --model rnnt \
    --checkpoint /tmp/parakeet-tdt-export \
    audio.wav
```

### Canary (transcription)

```bash
./nemo-transcribe \
    --model canary \
    --checkpoint /tmp/canary-export \
    --lang en \
    audio.wav
```

### Canary (translation)

```bash
./nemo-transcribe \
    --model canary \
    --checkpoint /tmp/canary-export \
    --lang en \
    --task translate \
    --tgt-lang de \
    audio.wav
```

### Multiple files

```bash
./nemo-transcribe --model parakeet --checkpoint /tmp/export file1.wav file2.mp3 file3.flac
```

Any audio format supported by ffmpeg is accepted.

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | `parakeet`, `rnnt`, or `canary` |
| `--checkpoint` | *(required)* | Path to checkpoint directory or `.nemo` file |
| `--ort-lib` | `$ORT_LIB_PATH` | Path to ONNX Runtime shared library |
| `--lang` | `en` | Source language (Canary only) |
| `--task` | `transcribe` | `transcribe` or `translate` (Canary only) |
| `--tgt-lang` | *(= lang)* | Target language for translation (Canary only) |
| `--pnc` | `true` | Punctuation & capitalisation (Canary only) |
| `--cuda` | `-1` | CUDA device ID (`-1` = CPU) |

### Output format

Transcripts are written to stdout; diagnostics and stats go to stderr.

```
audio: 18.36s  transcription: 4.21s  RTF: 0.229  words: 47

[audio.wav]
It takes a great deal of bravery to stand up to our enemies...
```

---

## Using as a library

```go
import (
    "github.com/dkaps125/nemo-go/checkpoint"
    "github.com/dkaps125/nemo-go/model/rnnt"
    "github.com/dkaps125/nemo-go/onnx"
)

checkpoint.SetORTLibraryPath("/usr/local/lib/libonnxruntime.dylib")

ck, _ := checkpoint.LoadRNNTCheckpoint("/tmp/parakeet-tdt-export")
m, _ := rnnt.Load(ck, onnx.DefaultSessionOptions())
defer m.Close()

segments := []model.AudioSegment{{PCM: pcm, SampleRate: 16000}}
transcripts, _ := m.Transcribe(ctx, segments, model.TranscribeOptions{})
fmt.Println(transcripts[0])
```

The same pattern works for `checkpoint.LoadParakeetCheckpoint` + `parakeet.Load` and `checkpoint.LoadCanaryCheckpoint` + `canary.Load`.

---

## Project layout

```
audio/          — log-mel spectrogram extraction (STFT, filterbank, normalization)
checkpoint/     — .nemo archive parser and checkpoint loaders
model/
  parakeet/     — Parakeet CTC model (greedy CTC decoding)
  canary/       — Canary AED model (autoregressive greedy decoding)
  rnnt/         — RNNT/TDT model (greedy TDT frame-stepping)
onnx/           — ONNX Runtime session wrapper (mixed int32/int64/float32 I/O)
tokenizer/      — SentencePiece tokenizer + Canary special tokens
internal/
  mathutil/     — Argmax, Softmax, LogSoftmax
cmd/
  nemo-transcribe/ — CLI entry point
scripts/
  export_onnx.py   — NeMo → ONNX export (supports all three model families)
```

---

## Troubleshooting

**`ORT library path not set`**
Set `ORT_LIB_PATH` or pass `--ort-lib`. The library must match the version in `go.mod` (`onnxruntime_go v1.27.0` requires ORT ≥ 1.19).

**`ONNX file not found`**
Run the export script first. For directory-based checkpoints the loader matches files by suffix (e.g. `*_encoder.onnx`), so all export outputs must be in the same directory.

**`ffmpeg: could not find codec parameters`**
The input file may be corrupted or an unsupported container. Verify with `ffprobe audio.wav`.

**Empty transcript**
Check that the exported `n_mels` in `*_rnnt_meta.json` matches the model's preprocessor config (should be set automatically by the export script).

**CUDA errors**
Ensure the ORT library was built with CUDA support and matches your CUDA/cuDNN version. Pass `--cuda 0` to enable GPU.
