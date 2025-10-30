
import numpy as np
import librosa
import soundfile as sf

ref_file = r"C:\Users\Admin\Downloads\archive\50_speakers_audio_data\Speaker_0013\Speaker_0013_00000.wav"  # B prototype
speaker_file = r"C:\Users\Admin\Desktop\custom speaker recogniser\vishnu_GAS_test_input.wav"  # A test input

def combine_audio_files(file1_path, file2_path, output_path="combined.wav"):
    """
    Load two audio files, add their waveforms together, and save as combined.wav
    """
    try:
        # Load the first audio file
        audio1, sr1 = librosa.load(file1_path, sr=None)
        print(f"Loaded {file1_path}: {len(audio1)} samples at {sr1} Hz")
        
        # Load the second audio file
        audio2, sr2 = librosa.load(file2_path, sr=None)
        print(f"Loaded {file2_path}: {len(audio2)} samples at {sr2} Hz")
        
        # Ensure both files have the same sample rate
        if sr1 != sr2:
            print(f"Sample rates differ: {sr1} vs {sr2}. Resampling to {sr1} Hz")
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        
        # Make both audio arrays the same length by padding the shorter one with zeros
        max_length = max(len(audio1), len(audio2))
        if len(audio1) < max_length:
            audio1 = np.pad(audio1, (0, max_length - len(audio1)))
        if len(audio2) < max_length:
            audio2 = np.pad(audio2, (0, max_length - len(audio2)))
        
        # Add the waveforms together
        combined_audio = audio1 + audio2
        
        # Normalize to prevent clipping (optional - keeps amplitude in [-1, 1] range)
        max_amplitude = np.max(np.abs(combined_audio))
        if max_amplitude > 1.0:
            combined_audio = combined_audio / max_amplitude
            print(f"Normalized audio to prevent clipping (original max: {max_amplitude:.3f})")
        
        # Save the combined audio
        sf.write(output_path, combined_audio, sr1)
        print(f"Combined audio saved as: {output_path}")
        print(f"Duration: {len(combined_audio) / sr1:.2f} seconds")
        
        return combined_audio, sr1
        
    except Exception as e:
        print(f"Error combining audio files: {e}")
        return None, None

if __name__ == "__main__":
    # Combine the two specified audio files
    combined_audio, sample_rate = combine_audio_files(ref_file, speaker_file)
    
    if combined_audio is not None:
        print("Audio combination completed successfully!")
    else:
        print("Audio combination failed!")