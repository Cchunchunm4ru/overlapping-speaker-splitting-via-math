import sounddevice as sd 
import wave
import torch
import os 
import numpy as np

CHANNELS = 1                  
fs = 16000                               
RECORD_SECONDS = 5            
OUTPUT_FILENAME = "chunchun.wav"

def record():
    print("the audio is being recorded....")
    recording = sd.rec(int(fs*RECORD_SECONDS),samplerate = fs,channels = 1)
    sd.wait()
    waveform = torch.tensor(recording.T, dtype=torch.float32).squeeze(0)
    print("recording is done....")
    print(waveform)
    recording_int16 = (recording * 32767).astype(np.int16)

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording_int16.tobytes())

    print(f"Audio saved as {OUTPUT_FILENAME}")

record()