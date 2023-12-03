import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sklearn

genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()

# for g in genres:
#     for filename in os.listdir(os.path.join('E:/Praca_inz/MusicGenreRecog/content/audio3sec',f"{g}")):
#         file = os.path.join(f'E:/Praca_inz/MusicGenreRecog/content/audio3sec/{g}', f'{filename}')

file = os.path.join('test\\disco.00004.wav')

signal,sr = librosa.load(file, sr=22050)

# using a power spectrum
chroma_d = librosa.feature.chroma_stft(signal, sr=sr)

plt.figure(figsize=(15, 4))
librosa.display.specshow(chroma_d, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Power spectrum chromagram')
plt.tight_layout()
plt.show()
# compute the spectral centroid for each frame in a signal
spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
spectral_centroids.shape

# compute the time variable for visualization
frames = range(len(spectral_centroids))
f_times = librosa.frames_to_time(frames)

# an auxiliar function to normalize the spectral centroid for visualization
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

plt.figure(figsize=(15,5))
plt.subplot(1, 1, 1)
librosa.display.waveshow(signal, sr=sr, alpha=0.4)
plt.plot(f_times, normalize(spectral_centroids), color='black', label='spectral centroids')
plt.ylabel('Hz')
plt.xticks([])
plt.legend()
plt.show()

# waveform
   # sr * T -> 22050 * 30
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

plt.plot(frequency, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency) / 2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# stft -> spectrogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCCs
MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()

