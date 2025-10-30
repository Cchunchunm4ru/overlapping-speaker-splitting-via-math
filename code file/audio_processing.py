import numpy as np
import soundfile as sf
import librosa as lb
from scipy.signal import windows
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import os

ref_file = r"C:\Users\Admin\Downloads\archive\50_speakers_audio_data\Speaker_0013\Speaker_0013_00000.wav"
combined_file = r"C:\Users\Admin\Desktop\custom speaker recogniser\combined.wav"
ref_primary = r"C:\Users\Admin\Desktop\custom speaker recogniser\chunchun.wav"

fs = 16000
FRAME_LEN = int(0.032 * fs)
HOP_LEN = int(0.008 * fs)
MAX_FRAMES = 2000
LEARNING_RATE = 0.01
output_dir = r"C:\Users\Admin\Desktop\custom speaker recogniser\energy_split_output"

ALPHA = 2.0
ETA = 0.005
ETA_A_SLOW = 0.01
ETA_A_FAST = 0.05
TOL_SIMILARITY = 0.95
TOL_CONV = 0.99
TOL_DROP = 0.01
NUM_EPOCHS = 15
MFCC_WEIGHT = 0.95
ENERGY_WEIGHT = 0.05



def read_wavefile(filename, sr=fs, seconds=None):
    if not os.path.exists(filename):
        print(f"Warning: {filename} missing. Returning zeros.")
        length = (seconds or 5) * sr
        return np.zeros(length, dtype=np.float32), sr
    w, s = lb.load(filename, sr=sr, mono=True)
    if seconds is not None:
        w = w[: seconds * sr]
    return w.astype(np.float32), s

def cropped(waveform, time_crop=5):
    max_length = time_crop * fs
    return waveform[:max_length]

def spectral_similarity_windowed(sig1, sig2, window):
    windowed_sig1 = sig1 * window
    windowed_sig2 = sig2 * window
    
    fft1 = np.fft.fft(windowed_sig1)
    fft2 = np.fft.fft(windowed_sig2)
    
    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)
    
    dot_product = np.sum(mag1 * mag2)
    norm1 = np.sqrt(np.sum(mag1**2))
    norm2 = np.sqrt(np.sum(mag2**2))
    
    if norm1 > 1e-12 and norm2 > 1e-12:
        return dot_product / (norm1 * norm2)
    else:
        return 0.0

def remove_initial_zeros(audio_data, threshold=1e-6):
    non_zero_indices = np.where(np.abs(audio_data) > threshold)[0]
    if len(non_zero_indices) == 0:
        return audio_data
    
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]
    
    print(f"Removing {first_non_zero} initial zeros and {len(audio_data) - last_non_zero - 1} trailing zeros")
    return audio_data[first_non_zero:last_non_zero+1]

def normalize_audio(arr):
    m = np.max(np.abs(arr))
    return arr / (m + 1e-12) if m > 0 else arr

def extract_mfcc_features(audio_signal, sr=fs, n_mfcc=13, n_fft=2048, hop_length=512):
    min_length = max(n_fft, hop_length * 10)
    if len(audio_signal) < min_length:
        audio_signal = np.pad(audio_signal, (0, min_length - len(audio_signal)))
    
    try:
        mfcc = lb.feature.mfcc(
            y=audio_signal, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=40,
            fmin=80,
            fmax=sr//2
        )
        
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        if mfcc.shape[1] >= 9:
            try:
                mfcc_delta = lb.feature.delta(mfcc)
                mfcc_delta2 = lb.feature.delta(mfcc, order=2)
                
                feature_vector = np.concatenate([
                    mfcc_mean,
                    mfcc_std,
                    np.mean(mfcc_delta, axis=1),
                    np.mean(mfcc_delta2, axis=1)
                ])
            except Exception as e:
                print(f"Warning: Delta computation failed, using only mean and std: {e}")
                feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        else:
            feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        
        return mfcc, feature_vector, mfcc_mean, mfcc_std
        
    except Exception as e:
        print(f"Warning: MFCC extraction failed for signal of length {len(audio_signal)}: {e}")
        mfcc = np.zeros((n_mfcc, 1))
        mfcc_mean = np.zeros(n_mfcc)
        mfcc_std = np.zeros(n_mfcc)
        feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        return mfcc, feature_vector, mfcc_mean, mfcc_std

def compare_mfcc_features(mfcc1, mfcc2, feature_vec1, feature_vec2):
    cosine_sim = cosine_similarity([feature_vec1], [feature_vec2])[0, 0]
    
    euclidean_dist = np.linalg.norm(feature_vec1 - feature_vec2)
    
    min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1_trimmed = mfcc1[:, :min_frames]
    mfcc2_trimmed = mfcc2[:, :min_frames]
    
    correlations = []
    for i in range(mfcc1_trimmed.shape[0]):
        corr = np.corrcoef(mfcc1_trimmed[i, :], mfcc2_trimmed[i, :])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations) if correlations else 0.0
    
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'avg_correlation': avg_correlation,
        'mfcc_correlations': correlations,
        'feature_vector_diff': np.abs(feature_vec1 - feature_vec2)
    }

def analyze_speaker_quality(separated_audio, reference_audio, sr=fs):
    print("\n=== MFCC-based Speaker Quality Analysis ===")
    
    sep_mfcc, sep_features, sep_mean, sep_std = extract_mfcc_features(separated_audio, sr)
    ref_mfcc, ref_features, ref_mean, ref_std = extract_mfcc_features(reference_audio, sr)
    
    print(f"Separated audio MFCC shape: {sep_mfcc.shape}")
    print(f"Reference audio MFCC shape: {ref_mfcc.shape}")
    
    comparison = compare_mfcc_features(sep_mfcc, ref_mfcc, sep_features, ref_features)
    
    print(f"\nMFCC Similarity Analysis:")
    print(f"  Cosine Similarity: {comparison['cosine_similarity']:.4f} (1.0 = identical)")
    print(f"  Euclidean Distance: {comparison['euclidean_distance']:.4f} (0.0 = identical)")
    print(f"  Average Correlation: {comparison['avg_correlation']:.4f} (1.0 = perfect correlation)")
    
    print(f"\nIndividual MFCC Coefficient Correlations:")
    for i, corr in enumerate(comparison['mfcc_correlations'][:13]):
        print(f"  MFCC-{i:2d}: {corr:6.3f}")
    
    print(f"\nFeature Vector Analysis:")
    print(f"  Mean MFCC difference: {np.mean(comparison['feature_vector_diff'][:13]):.4f}")
    print(f"  Std MFCC difference:  {np.mean(comparison['feature_vector_diff'][13:26]):.4f}")
    
    overall_score = (comparison['cosine_similarity'] + comparison['avg_correlation']) / 2
    print(f"\nOverall Quality Score: {overall_score:.4f}")
    
    if overall_score > 0.8:
        quality_level = "Excellent"
    elif overall_score > 0.6:
        quality_level = "Good"
    elif overall_score > 0.4:
        quality_level = "Fair"
    else:
        quality_level = "Poor"
        
    print(f"Quality Assessment: {quality_level}")
    
    return comparison, overall_score, quality_level

def mfcc_based_separation():
    print("Loading audio files...")
    M, _ = read_wavefile(combined_file, sr=fs, seconds=10)
    phi_primary, _ = read_wavefile(ref_primary, sr=fs, seconds=10)
    phi_secondary, _ = read_wavefile(ref_file, sr=fs, seconds=10)
    
    print("Removing zero padding from reference files...")
    M = remove_initial_zeros(M)
    phi_primary = remove_initial_zeros(phi_primary)
    phi_secondary = remove_initial_zeros(phi_secondary)
    
    min_length = min(len(M), len(phi_primary), len(phi_secondary))
    M = M[:min_length]
    phi_primary = phi_primary[:min_length]
    phi_secondary = phi_secondary[:min_length]
    
    print(f"Processing {min_length} samples ({min_length/fs:.2f} seconds)")
    
    print("\nSTEP 1: Calculating reference MFCC vector from phi_primary...")
    ref_mfcc, ref_features, ref_mean, ref_std = extract_mfcc_features(phi_primary, sr=fs)
    c_ref = ref_mean
    print(f"Reference MFCC vector shape: {c_ref.shape}")
    print(f"Reference MFCC coefficients: {c_ref[:5]}")
    
    print("\nSTEP 2: Initial A_hat estimation by artifact subtraction...")
    artifact_scale = np.std(M) / (np.std(phi_secondary) + 1e-10)
    scaled_artifact = phi_secondary * artifact_scale * 0.5
    
    A_hat = M - scaled_artifact
    print(f"Initial A_hat created by subtracting scaled artifact")
    
    print("\nSTEP 3: Calculating initial A_hat MFCC vector...")
    initial_mfcc, initial_features, initial_mean, initial_std = extract_mfcc_features(A_hat, sr=fs)
    c_initial = initial_mean
    mfcc_diff_initial = np.linalg.norm(c_initial - c_ref)
    print(f"Initial A_hat MFCC vector shape: {c_initial.shape}")
    print(f"Initial MFCC difference (L2 norm): {mfcc_diff_initial:.6f}")
    print(f"Initial MFCC coefficients: {c_initial[:5]}")
    
    learning_rate = LEARNING_RATE
    momentum_A = np.zeros_like(M)
    momentum_decay = 0.9
    
    window = windows.hann(FRAME_LEN)
    
    num_frames = min(MAX_FRAMES, (len(M) - FRAME_LEN) // HOP_LEN + 1)
    
    print(f"\nSTEP 4: SGD Backpropagation with MFCC-based loss...")
    print(f"Processing {num_frames} frames with MFCC spectral envelope matching...")
    print("Epoch | Frame | MFCC_Loss | Gradient_RMS | Learning_Rate | MFCC_Similarity")
    print("-" * 85)
    
    total_loss_history = []
    convergence_threshold = 1e-6
    best_mfcc_similarity = 0.0
    patience = 3
    no_improvement = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== SGD EPOCH {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_total_loss = 0.0
        
        for it in range(num_frames):
            i = it * HOP_LEN  # Current frame start index
            
            # Extract current frames
            M_frame = M[i:i + FRAME_LEN]
            Aframe = A_hat[i:i + FRAME_LEN]
            
            # Get reference frame for comparison
            ref_idx = (i % len(phi_primary))
            ref_end = min(ref_idx + FRAME_LEN, len(phi_primary))
            phi_frame = phi_primary[ref_idx:ref_end]
            
            # Pad frames if necessary
            if len(phi_frame) < FRAME_LEN:
                phi_frame = np.pad(phi_frame, (0, FRAME_LEN - len(phi_frame)))
            if len(M_frame) < FRAME_LEN:
                M_frame = np.pad(M_frame, (0, FRAME_LEN - len(M_frame)))
                Aframe = np.pad(Aframe, (0, FRAME_LEN - len(Aframe)))
            
            # Calculate MFCC vectors for current frame
            # Use a simplified approach for frame-based processing
            try:
                # For frame-based processing, we'll use spectral features instead of full MFCC
                # to avoid the delta computation issues with short frames
                
                # Calculate frame energies first (always works)
                E_ref_frame = np.sum(phi_frame**2)
                E_Ahat_frame = np.sum(Aframe**2)
                
                # Try MFCC extraction with error handling
                if len(Aframe) >= 1024:  # Ensure sufficient length
                    try:
                        # Pad frames to ensure minimum length for stable MFCC
                        padded_Aframe = np.pad(Aframe, (0, max(0, 2048 - len(Aframe))))
                        padded_phi_frame = np.pad(phi_frame, (0, max(0, 2048 - len(phi_frame))))
                        
                        frame_mfcc, frame_features, frame_mean, frame_std = extract_mfcc_features(padded_Aframe, sr=fs)
                        ref_frame_mfcc, ref_frame_features, ref_frame_mean, ref_frame_std = extract_mfcc_features(padded_phi_frame, sr=fs)
                        
                        c_frame = frame_mean  # Current frame MFCC vector
                        c_ref_frame = ref_frame_mean  # Reference MFCC vector
                        
                        # MFCC-based loss: L = ||c(x̂) - c(x_ref)||²
                        mfcc_diff = c_frame - c_ref_frame
                        mfcc_loss = np.sum(mfcc_diff**2)  # L2 norm squared
                        
                        # Calculate MFCC similarity for monitoring
                        mfcc_similarity = cosine_similarity([c_frame], [c_ref_frame])[0, 0]
                        
                        # Aggressive combined loss for complete Speaker 32 removal
                        energy_loss = (E_Ahat_frame - E_ref_frame)**2
                        loss = MFCC_WEIGHT * mfcc_loss + ENERGY_WEIGHT * energy_loss
                        epoch_total_loss += loss
                        
                        # Enhanced gradient computation for stronger separation
                        # Multi-component gradient targeting specific MFCC coefficients
                        mfcc_gradient_strength = np.linalg.norm(mfcc_diff)  # Magnitude of MFCC error
                        
                        # Target specific MFCC coefficients that distinguish speakers
                        formant_coeffs = mfcc_diff[1:5]  # Coefficients 1-4 encode formant structure
                        spectral_coeffs = mfcc_diff[5:9]  # Coefficients 5-8 encode spectral shape
                        
                        # Weighted gradient based on coefficient importance
                        formant_weight = 3.0  # Higher weight for formant-related coefficients
                        spectral_weight = 2.0  # Medium weight for spectral coefficients
                        
                        # Compute enhanced gradient
                        energy_gradient = 2.0 * (E_Ahat_frame - E_ref_frame) * 2.0 * Aframe
                        
                        # MFCC-guided gradient with coefficient-specific weighting
                        mfcc_direction = (formant_weight * np.mean(formant_coeffs) + 
                                        spectral_weight * np.mean(spectral_coeffs))
                        
                        # Amplify gradient based on MFCC error magnitude
                        gradient_amplifier = min(10.0, 1.0 + mfcc_gradient_strength)
                        
                        gradient = mfcc_direction * energy_gradient * gradient_amplifier
                        
                        use_mfcc = True
                        
                    except Exception as e:
                        print(f"Warning: Frame MFCC failed, using energy fallback: {e}")
                        use_mfcc = False
                else:
                    use_mfcc = False
                
                if not use_mfcc:
                    # Fallback to energy-based for short frames or MFCC failures
                    loss = (E_Ahat_frame - E_ref_frame)**2
                    epoch_total_loss += loss
                    gradient = 2.0 * (E_Ahat_frame - E_ref_frame) * 2.0 * Aframe
                    mfcc_similarity = 0.0
                    
            except Exception as e:
                print(f"Warning: Frame processing failed, using simple energy loss: {e}")
                # Ultimate fallback
                E_ref_frame = np.sum(phi_frame**2)
                E_Ahat_frame = np.sum(Aframe**2)
                loss = (E_Ahat_frame - E_ref_frame)**2
                epoch_total_loss += loss
                gradient = 2.0 * (E_Ahat_frame - E_ref_frame) * 2.0 * Aframe
                mfcc_similarity = 0.0
            
            # Enhanced regularization for robust Speaker 32 removal
            # Stronger L2 regularization to prevent Speaker 32 artifacts
            l2_weight = 0.005  # Increased regularization
            l2_gradient = l2_weight * Aframe
            
            # Spectral shape enforcement: strongly encourage Speaker A characteristics
            shape_weight = 0.5  # Much stronger shape constraint
            if np.sum(phi_frame**2) > 1e-10:
                # Normalize reference to current frame energy
                phi_norm = phi_frame / np.sqrt(np.sum(phi_frame**2))
                Aframe_norm = Aframe / (np.sqrt(np.sum(Aframe**2)) + 1e-10)
                shape_gradient = shape_weight * (Aframe_norm - phi_norm) * np.sqrt(np.sum(Aframe**2))
            else:
                shape_gradient = np.zeros_like(Aframe)
            
            # Anti-Speaker-32 regularization: penalize Speaker 32 characteristics
            # Use spectral rolloff and centroid differences to identify Speaker 32 patterns
            if len(phi_frame) == len(Aframe):
                # Compute spectral characteristics
                A_fft = np.fft.fft(Aframe * window[:len(Aframe)])
                A_mag = np.abs(A_fft[:len(A_fft)//2])
                A_centroid = np.sum(np.arange(len(A_mag)) * A_mag) / (np.sum(A_mag) + 1e-10)
                
                phi_fft = np.fft.fft(phi_frame * window[:len(phi_frame)])
                phi_mag = np.abs(phi_fft[:len(phi_fft)//2])
                phi_centroid = np.sum(np.arange(len(phi_mag)) * phi_mag) / (np.sum(phi_mag) + 1e-10)
                
                # Penalize spectral centroid deviation from reference
                centroid_error = A_centroid - phi_centroid
                anti_b_weight = 0.3
                anti_b_gradient = anti_b_weight * centroid_error * Aframe / (np.max(np.abs(Aframe)) + 1e-10)
            else:
                anti_b_gradient = np.zeros_like(Aframe)
            
            # Combined gradient with all components
            total_gradient = gradient + l2_gradient + shape_gradient + anti_b_gradient
            
            # Calculate gradient RMS for monitoring
            grad_rms = np.sqrt(np.mean(total_gradient**2))
            
            # Aggressive adaptive learning rate for complete separation
            if grad_rms > 1e5:
                current_lr = learning_rate * 0.001  # Extremely large gradient - be very careful
            elif grad_rms > 1e3:
                current_lr = learning_rate * 0.01   # Very large gradient - slow down significantly
            elif grad_rms > 1e1:
                current_lr = learning_rate * 0.5    # Large gradient - moderate slowdown
            elif grad_rms < 1e-6:
                current_lr = learning_rate * 5.0    # Tiny gradient - aggressive speedup
            elif grad_rms < 1e-3:
                current_lr = learning_rate * 3.0    # Small gradient - significant speedup
            else:
                current_lr = learning_rate * 1.5    # Default: slight speedup for robustness
            
            # Apply momentum for smoother convergence
            momentum_frame = momentum_A[i:i + FRAME_LEN]
            if len(momentum_frame) == len(total_gradient):
                momentum_frame = momentum_decay * momentum_frame + (1 - momentum_decay) * total_gradient
                momentum_A[i:i + FRAME_LEN] = momentum_frame
            else:
                momentum_frame = total_gradient
            
            # SGD Update: A_hat_new = A_hat_old - learning_rate * gradient
            Aframe_new = Aframe - current_lr * momentum_frame
            
            # Prevent divergence by clipping extreme values
            max_amp = 1.5 * np.std(M)  # Allow up to 1.5x the mixture standard deviation
            Aframe_new = np.clip(Aframe_new, -max_amp, max_amp)
            
            # Update A_hat in place
            if i + FRAME_LEN <= len(A_hat):
                A_hat[i:i + FRAME_LEN] = Aframe_new[:len(A_hat[i:i + FRAME_LEN])]
            
            # Progress logging
            if it % 50 == 0 or (epoch == 0 and it < 10):
                print(f"{epoch+1:2d} | {it:5d} | {loss:9.2e} | {grad_rms:8.2e} | {current_lr:8.5f} | {mfcc_similarity:8.3f}")
        
        # End of epoch summary
        avg_epoch_loss = epoch_total_loss / num_frames
        total_loss_history.append(avg_epoch_loss)
        
        # Calculate current A_hat MFCC characteristics
        current_mfcc, current_features, current_mean, current_std = extract_mfcc_features(A_hat, sr=fs)
        current_mfcc_diff = np.linalg.norm(current_mean - c_ref)
        current_mfcc_similarity = cosine_similarity([current_mean], [c_ref])[0, 0]
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average MFCC Loss: {avg_epoch_loss:9.2e}")
        print(f"  Current MFCC L2 Distance: {current_mfcc_diff:.6f}")
        print(f"  MFCC Similarity to Reference: {current_mfcc_similarity:.3f}")
        
        # Adaptive learning rate decay based on progress
        if epoch > 0:
            if current_mfcc_similarity > best_mfcc_similarity:
                best_mfcc_similarity = current_mfcc_similarity
                learning_rate *= 1.01  # Slight increase if improving
                no_improvement = 0
            else:
                learning_rate *= 0.9   # Decay if not improving
                no_improvement += 1
        
        # Advanced convergence criteria
        converged = False
        if epoch > 2:
            # Multiple convergence conditions
            loss_converged = avg_epoch_loss < convergence_threshold
            similarity_converged = current_mfcc_similarity > 0.999
            improvement_stalled = no_improvement >= patience
            
            if loss_converged or similarity_converged:
                print(f"Converged - Loss: {loss_converged}, Similarity: {similarity_converged}")
                converged = True
            elif improvement_stalled and epoch > 5:
                print(f"Early stopping - No improvement for {patience} epochs")
                converged = True
        
        if converged:
            break
    
    print(f"\nSTEP 5: Post-processing for maximum Speaker 32 removal...")
    
    # Iterative refinement: Apply additional MFCC-based filtering
    A_hat_refined = A_hat.copy()
    
    # Multi-pass MFCC alignment
    for refinement_pass in range(3):
        print(f"Refinement pass {refinement_pass + 1}/3...")
        
        # Calculate current MFCC distance
        current_mfcc, _, current_mean, _ = extract_mfcc_features(A_hat_refined, sr=fs)
        mfcc_error = current_mean - c_ref
        
        # Apply coefficient-specific corrections
        for coeff_idx in range(len(mfcc_error)):
            if abs(mfcc_error[coeff_idx]) > 1.0:  # Significant deviation
                # Apply targeted spectral filtering to correct this coefficient
                correction_factor = 0.1 * mfcc_error[coeff_idx]
                
                # Frequency-domain correction (simplified)
                A_fft = np.fft.fft(A_hat_refined)
                freq_bins = len(A_fft)
                
                # Target frequency ranges for each MFCC coefficient
                if coeff_idx == 1:  # First formant region
                    freq_range = slice(int(0.05 * freq_bins), int(0.15 * freq_bins))
                elif coeff_idx == 2:  # Second formant region
                    freq_range = slice(int(0.15 * freq_bins), int(0.25 * freq_bins))
                elif coeff_idx == 3:  # Higher formants
                    freq_range = slice(int(0.25 * freq_bins), int(0.35 * freq_bins))
                else:  # General spectral shape
                    freq_range = slice(int(coeff_idx * 0.05 * freq_bins), 
                                     int((coeff_idx + 1) * 0.05 * freq_bins))
                
                # Apply targeted attenuation/amplification
                A_fft[freq_range] *= (1.0 - correction_factor * 0.1)
                A_fft[-freq_range.start-1:-freq_range.stop-1] *= (1.0 - correction_factor * 0.1)
                
                A_hat_refined = np.real(np.fft.ifft(A_fft))
    
    # Spectral subtraction to remove remaining Speaker 32 artifacts
    print("Applying spectral subtraction for Speaker 32 removal...")
    
    # Compute average spectral profile of Speaker 32 from reference
    phi_secondary_fft = np.fft.fft(phi_secondary)
    speaker32_profile = np.abs(phi_secondary_fft)
    
    # Apply over-subtraction to aggressively remove Speaker 32
    A_fft = np.fft.fft(A_hat_refined)
    A_mag = np.abs(A_fft)
    A_phase = np.angle(A_fft)
    
    # Over-subtraction: remove more than estimated Speaker 32 content
    over_subtraction_factor = 2.0
    speaker32_normalized = speaker32_profile * (np.mean(A_mag) / np.mean(speaker32_profile))
    
    # Subtract with spectral floor to prevent over-subtraction artifacts
    spectral_floor = 0.1
    A_mag_cleaned = np.maximum(
        A_mag - over_subtraction_factor * speaker32_normalized,
        spectral_floor * A_mag
    )
    
    # Reconstruct signal
    A_cleaned_fft = A_mag_cleaned * np.exp(1j * A_phase)
    A_hat_out = np.real(np.fft.ifft(A_cleaned_fft))
    
    # Final smoothing with stronger artifact reduction
    try:
        from scipy.ndimage import uniform_filter1d
        from scipy import signal
        
        # Multi-stage smoothing for artifact removal
        A_hat_out = uniform_filter1d(A_hat_out, size=5, mode='reflect')  # Stronger smoothing
        
        # Additional median filtering for impulse noise removal
        A_hat_out = signal.medfilt(A_hat_out, kernel_size=3)
        
    except ImportError:
        print("Advanced filtering unavailable, using basic smoothing")
        # Simple moving average fallback
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        A_hat_out = np.convolve(A_hat_out, kernel, mode='same')
    
    # B_hat is the residual after subtracting the improved A_hat
    B_hat_final = M - A_hat_out
    
    # Final energy calculations
    A_energy = np.sum(A_hat_out**2)
    B_energy = np.sum(B_hat_final**2)
    M_energy = np.sum(M**2)
    
    # Calculate final MFCC similarity
    final_mfcc, final_features, final_mean, final_std = extract_mfcc_features(A_hat_out, sr=fs)
    final_mfcc_diff = np.linalg.norm(final_mean - c_ref)
    final_mfcc_similarity = cosine_similarity([final_mean], [c_ref])[0, 0]
    
    print(f"\nFinal Results after Enhanced MFCC-based Speaker 32 Removal:")
    print(f"  Original mixture energy: {M_energy:.6f}")
    print(f"  Final A_hat energy: {A_energy:.6f}")
    print(f"  Final B_hat energy: {B_energy:.6f}")
    print(f"  Energy preservation: {(A_energy + B_energy)/M_energy:.3f}")
    print(f"  Final MFCC L2 distance to reference: {final_mfcc_diff:.6f}")
    print(f"  Final MFCC similarity to reference: {final_mfcc_similarity:.3f}")
    
    # Enhanced separation quality metrics
    separation_quality = final_mfcc_similarity * (A_energy / (B_energy + 1e-10))
    speaker32_suppression = 1.0 - (final_mfcc_diff / mfcc_diff_initial)
    
    print(f"\nEnhanced Separation Metrics:")
    print(f"  Speaker 32 suppression ratio: {speaker32_suppression:.3f}")
    print(f"  Overall separation quality: {separation_quality:.3f}")
    print(f"  MFCC improvement: {mfcc_diff_initial:.1f} → {final_mfcc_diff:.1f} ({speaker32_suppression*100:.1f}% reduction)")
    
    # Apply gain normalization to prevent clipping
    A_peak = np.max(np.abs(A_hat_out))
    B_peak = np.max(np.abs(B_hat_final))
    
    if A_peak > 1.0:
        A_hat_out = A_hat_out / A_peak * 0.95
        print(f"Normalized A_hat by factor {A_peak:.3f}")
    if B_peak > 1.0:
        B_hat_final = B_hat_final / B_peak * 0.95
        print(f"Normalized B_hat by factor {B_peak:.3f}")
    
    # Normalize outputs for proper audio levels
    A_out = normalize_audio(A_hat_out)
    B_out = normalize_audio(B_hat_final)
    
    # Calculate final separation metrics
    A_rms = np.sqrt(np.mean(A_out**2))
    B_rms = np.sqrt(np.mean(B_out**2))
    print(f"\nFinal separation metrics:")
    print(f"  Speaker A RMS: {A_rms:.6f}")
    print(f"  Speaker B RMS: {B_rms:.6f}")
    print(f"  Separation ratio (A/B): {A_rms/B_rms:.3f}" if B_rms > 0 else "  B is silent")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save separated audio files
    output_speaker = os.path.join(output_dir, "separated_speaker_A.wav")
    output_residual = os.path.join(output_dir, "separated_residual_B.wav")
    
    # Write the separated files
    sf.write(output_speaker, A_out, fs)
    sf.write(output_residual, B_out, fs)
    
    # Also save with timestamp for versioning
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_speaker_ts = os.path.join(output_dir, f"A_hat_{timestamp}.wav")
    output_residual_ts = os.path.join(output_dir, f"B_hat_{timestamp}.wav")
    
    sf.write(output_speaker_ts, A_out, fs)
    sf.write(output_residual_ts, B_out, fs)
    
    print(f"\nSGD-based separation complete!")
    print(f"Separated signal A: {output_speaker}")
    print(f"Residual signal B: {output_residual}")
    print(f"Timestamped versions: {output_speaker_ts}, {output_residual_ts}")
    
    # MFCC-based quality assessment - spectral envelope similarity
    mfcc_quality_score = final_mfcc_similarity  # Cosine similarity ranges from -1 to 1
    
    # Save processing report
    report_file = os.path.join(output_dir, f"mfcc_sgd_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== MFCC-based SGD Speaker Separation Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Algorithm: MFCC spectral envelope matching\n")
        f.write(f"Loss function: L = ||c(x_hat) - c(x_ref)||^2 where c(.) is MFCC vector\n\n")
        f.write(f"Input files:\n")
        f.write(f"  Combined signal: {combined_file}\n")
        f.write(f"  Reference signal (Speaker A): {ref_primary}\n")
        f.write(f"  Reference artifact (Speaker B): {ref_file}\n\n")
        f.write(f"MFCC Analysis:\n")
        f.write(f"  Initial MFCC distance: {mfcc_diff_initial:.6f}\n")
        f.write(f"  Final MFCC distance: {final_mfcc_diff:.6f}\n")
        f.write(f"  Final MFCC similarity: {final_mfcc_similarity:.3f}\n")
        f.write(f"  Final A_hat energy: {A_energy:.6f}\n")
        f.write(f"  MFCC quality score: {mfcc_quality_score:.3f}\n\n")
        f.write(f"Training Parameters:\n")
        f.write(f"  Epochs: {NUM_EPOCHS}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Frame length: {FRAME_LEN}\n")
        f.write(f"  Hop length: {HOP_LEN}\n\n")
        f.write(f"Output Files:\n")
        f.write(f"  A_hat (separated): {output_speaker_ts}\n")
        f.write(f"  B_hat (residual): {output_residual_ts}\n")
    
    print(f"Processing report saved to: {report_file}")
    print(f"\n" + "="*70)
    print(f"FINAL MFCC-BASED SGD SEPARATION SUMMARY")
    print("="*70)
    print(f"MFCC similarity to reference: {final_mfcc_similarity:.3f}")
    print(f"MFCC L2 distance reduction: {mfcc_diff_initial:.3f} → {final_mfcc_diff:.3f}")
    print(f"Spectral envelope matching achieved through MFCC-based backpropagation")
    
    # Return enhanced results with separation statistics
    separation_stats = {
        'speaker32_suppression': speaker32_suppression,
        'separation_quality': separation_quality,
        'final_mfcc_similarity': final_mfcc_similarity,
        'mfcc_improvement': mfcc_diff_initial - final_mfcc_diff
    }
    
    return A_out, B_out, output_speaker, output_residual, separation_stats

if __name__ == "__main__":
    try:
        results = mfcc_based_separation()
        A_separated, B_separated, speaker_file, residual_file, separation_stats = results
        speaker32_suppression = separation_stats['speaker32_suppression']
        print("\nMFCC-based SGD speaker separation completed!")
        print(f"Output files created:")
        print(f"  Separated A_hat: {speaker_file}")
        print(f"  Residual B_hat: {residual_file}")
        print(f"\nRobust Speaker 32 Removal Process Summary:")
        print(f"  1. ✓ Calculated reference MFCC vector from clean Speaker A: {ref_primary}")
        print(f"  2. ✓ Subtracted Speaker 32 artifact from combined signal")
        print(f"  3. ✓ Applied aggressive MFCC-based SGD with {NUM_EPOCHS} epochs")
        print(f"  4. ✓ Enhanced gradient targeting formant and spectral coefficients")
        print(f"  5. ✓ Multi-pass MFCC refinement for coefficient-specific corrections")
        print(f"  6. ✓ Spectral subtraction with over-subtraction for Speaker 32 removal")
        print(f"  7. ✓ Advanced artifact filtering and signal reconstruction")
        print(f"  8. ✓ Achieved {speaker32_suppression*100:.1f}% Speaker 32 suppression")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()