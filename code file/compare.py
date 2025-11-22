import matplotlib.pyplot as plt
import librosa
import numpy as np

output = r"C:\Users\Admin\Desktop\projects\custom speaker recogniser\output_mfcc\verified_extracted_wave.wav"
combined_file = r"C:\Users\Admin\Desktop\projects\custom speaker recogniser\combined.wav"

# Load only first 5 seconds of audio
max_duration = 5  # seconds

y1, sr1 = librosa.load(output, sr=None, duration=max_duration)
y2, sr2 = librosa.load(combined_file, sr=None, duration=max_duration)

# Create time axes
t1 = np.linspace(0, len(y1) / sr1, num=len(y1))
t2 = np.linspace(0, len(y2) / sr2, num=len(y2))

# Plot side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(t1, y1)
axs[0].set_title('Output File')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')

axs[1].plot(t2, y2)
axs[1].set_title('Combined File')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()