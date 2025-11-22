import numpy as np
import soundfile as sf
import librosa as lb
import os
import sympy as sp

FS = 16000
FRAME_SIZE = int(0.010 * FS)
MFCC_STEP = int(0.100 * FS)

ref_file = r"C:\Users\Admin\Downloads\archive (3)\50_speakers_audio_data\Speaker_0007\Speaker_0007_00002.wav"
combined_file = r"C:\Users\Admin\Desktop\projects\custom speaker recogniser\combined.wav"
ref_primary = r"C:\Users\Admin\Desktop\projects\custom speaker recogniser\chunchun.wav"

def adam_update(param, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

def frame_based_gradient(estimated_frame, ref_frame):
    E_ref = np.sum(ref_frame**2)
    E_estimated = np.sum(estimated_frame**2)
    dL_d_estimated_frame = 4 * (E_estimated - E_ref) * estimated_frame 
    return dL_d_estimated_frame, E_ref, E_estimated

def calculate_mfcc_similarity(segment_estimated, segment_ref, n_mfcc=13):
    if len(segment_estimated) < MFCC_STEP or len(segment_ref) < MFCC_STEP:
        return 0.0
    try:
        mfcc_estimated = lb.feature.mfcc(y=segment_estimated, sr=FS, n_mfcc=n_mfcc)
        mfcc_ref = lb.feature.mfcc(y=segment_ref, sr=FS, n_mfcc=n_mfcc)
        mfcc_avg_estimated = np.mean(mfcc_estimated, axis=1)
        mfcc_avg_ref = np.mean(mfcc_ref, axis=1)
        dot_product = np.dot(mfcc_avg_estimated, mfcc_avg_ref)
        norm_estimated = np.linalg.norm(mfcc_avg_estimated)
        norm_ref = np.linalg.norm(mfcc_avg_ref)
        if norm_estimated == 0 or norm_ref == 0:
            return 0.0
        similarity = dot_product / (norm_estimated * norm_ref)
        return similarity
    except Exception as e:
        print(f"MFCC calculation failed: {e}")
        return 0.0

def energy_and_mfcc_separation_frame(combined_path, ref_a_path, max_iter=50, energy_threshold=0.94):
    mixture, _ = lb.load(combined_path, sr=FS)
    ref_a, _ = lb.load(ref_a_path, sr=FS)
    length = min(len(mixture), len(ref_a))
    x = mixture[:length]
    phi = ref_a[:length]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    output_separated = np.zeros_like(x)
    mfcc_verified = False
    print(f"Starting frame-by-frame separation... Total samples: {length}, Frame Size: {FRAME_SIZE}")
    for i in range(0, length - FRAME_SIZE, FRAME_SIZE):
        start_idx = i
        end_idx = i + FRAME_SIZE
        mix_frame = x[start_idx:end_idx]
        ref_frame = phi[start_idx:end_idx]
        estimated_a_frame = mix_frame - ref_frame
        m_frame = m[start_idx:end_idx]
        v_frame = v[start_idx:end_idx]
        iter_count = 0
        while iter_count < max_iter:
            grad, E_ref_frame, E_estimated_frame = frame_based_gradient(estimated_a_frame, ref_frame)
            corr = 1 - abs(E_ref_frame - E_estimated_frame)
            if corr >= energy_threshold:
                break
            estimated_a_frame, m_frame, v_frame = adam_update(
                estimated_a_frame, grad, m_frame, v_frame, iter_count + 1
            )
            iter_count += 1
        m[start_idx:end_idx] = m_frame
        v[start_idx:end_idx] = v_frame
        final_separated_frame = estimated_a_frame
        if i % MFCC_STEP == 0:
            mfcc_seg_end = min(i + MFCC_STEP, length)
            estimated_segment = output_separated[i:mfcc_seg_end]
            ref_segment = phi[i:mfcc_seg_end]
            mfcc_sim = calculate_mfcc_similarity(estimated_segment, ref_segment)
            if mfcc_sim < 0.965:
                output_separated[i:mfcc_seg_end] = 0.0
                mfcc_verified = False
            else:
                mfcc_verified = True
        if mfcc_verified:
            output_separated[start_idx:end_idx] = final_separated_frame
        else:
            output_separated[start_idx:end_idx] = 0.0
    output_dir = r"C:\Users\Admin\Desktop\projects\custom speaker recogniser\output_mfcc"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "verified_extracted_wave.wav")
    sf.write(output_path, output_separated, FS)
    print(f"\n--- Process Complete ---")
    print(f"Final audio length: {len(output_separated)} samples.")
    print(f"Saved MFCC-verified extracted waveform to {output_path}")

if __name__ == "__main__":
    energy_and_mfcc_separation_frame(combined_file, ref_primary, max_iter=50, energy_threshold=0.94)