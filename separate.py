import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

HF_TOKEN = "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"  # Replace with your token

INPUT_FOLDER = "raw"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Model load ho raha hai...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)
print(f"Using device: {device}")

files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.m4a', '.mp3', '.wav', '.aac'))]
print(f"Total {len(files)} files mili hain.\n")

for idx, filename in enumerate(files, 1):
    file_path = os.path.join(INPUT_FOLDER, filename)
    base_name = os.path.splitext(filename)[0]
    print(f"[{idx}/{len(files)}] Processing: {filename}")
    
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_wav = f"temp_{base_name}.wav"
        audio.export(temp_wav, format="wav")
        
        diarization = pipeline(temp_wav, num_speakers=2)
        
        speaker_tracks = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) < 1.2:
                continue
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]
            
            if speaker not in speaker_tracks:
                speaker_tracks[speaker] = AudioSegment.empty()
            speaker_tracks[speaker] += segment + AudioSegment.silent(duration=250)
        
        file_out_dir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(file_out_dir, exist_ok=True)
        
        for speaker, track in speaker_tracks.items():
            if len(track) > 2000:
                out_path = os.path.join(file_out_dir, f"{speaker}.wav")
                track.export(out_path, format="wav")
        
        os.remove(temp_wav)
        print("   ✅ Done")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n🎉 Complete! 'output' folder check karo.")
