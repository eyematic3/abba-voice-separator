# Abba Voice Separator

Automatic speaker diarization to extract one speaker's voice from mixed call recordings using **pyannote.audio**.

## Requirements
- Python 3.11 (recommended)
- FFmpeg installed and added to PATH
- Hugging Face account with access to pyannote models
- Hugging Face Read token

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/eyematic3/abba-voice-separator.git
   cd abba-voice-separator
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. Install packages:
   ```bash
   pip install --upgrade pip
   pip install torch torchaudio
   pip install pyannote.audio
   pip install pydub
   ```

4. Create folders:
   - `raw/` → put your call recordings here
   - `output/` → separated voices will be saved here

5. Edit `separate.py` and put your Hugging Face token:
   ```python
   HF_TOKEN = "hf_your_token_here"
   ```

6. Run:
   ```bash
   python separate.py
   ```

## Notes
- First run will download the model (takes time).
- Use `num_speakers=2` for normal phone calls.
- Check `output/` folder after processing.
- Recommended: Test with 5-10 files first.

## Hugging Face Access
Accept the terms on these pages:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1
