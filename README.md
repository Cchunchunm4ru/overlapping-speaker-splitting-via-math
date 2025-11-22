# 🎧 Overlapping Speaker Splitting via Math

> A mathematical and hybrid-signal approach to separating overlapping speakers without deep learning — inspired by energy constraints and spectral correlation.

---

## 🧠 Overview

Overlapping speech, where multiple speakers’ signals align into a single mixed waveform, remains one of the toughest challenges in real-time audio processing — especially under **low-latency** requirements.

This project explores an **energy-based mathematical approach** to separate mixed audio into individual sources using **gradient updates derived from physical energy constraints**.  
Later refinements add **MFCC-based perceptual correction** for improved quality.

---

## 🔬 Core Idea

Each speaker produces sound with a **unique energy signature**.  
If we can extract frame-wise energy and correlation patterns from a mixed signal, then **backpropagate an energy loss** onto an estimated random waveform, we can iteratively approximate each speaker’s component.

Mathematically:

\[
L = (E_x - E_{\hat{x}})^2
\]

\[
\frac{dL}{d\hat{A}(t)} = 2(E_{\hat{x}} - E_x)(2\hat{A}(t) - \phi(t))
\]

Where:
- \(E_x\) → energy of the reference (target) signal  
- \(E_{\hat{x}}\) → energy of the current estimated signal  
- \(\hat{A}(t)\) → estimated amplitude over time  
- \(\phi(t)\) → phase or window modulation term  

---

## ⚙️ Implementation Details

The implementation expands on the base math with:

| Component | Purpose |
|------------|----------|
| **Frame-based gradient updates** | Compute per-frame energy loss & adjust amplitude |
| **MFCC correlation term** | Aligns spectral envelopes (perceptual quality) |
| **L2 + Shape regularization** | Prevents artifact buildup |
| **Spectral centroid correction** | Adjusts tonal balance per iteration |
| **Post-optimization cleanup** | MFCC re-matching + spectral subtraction |
| **Energy normalization** | Keeps output in consistent amplitude range |

The system iterates over frames of the mixed audio, computes both **energy and MFCC losses**, updates the estimate using a **gradient-descent rule**, and refines it via a lightweight spectral filter.

---
## OUTPUT
<img width="1200" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/f32cfd08-7b82-4ca7-9071-0b284f3dcf63" />

## Derivation layout 


<img width="960" height="1280" alt="image" src="https://github.com/user-attachments/assets/428f0ff7-1906-4b00-a579-0977dfe68aba" />

