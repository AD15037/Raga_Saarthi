import numpy as np
import matplotlib.pyplot as plt

import librosa

filename = 'bhairavi30.wav'                 # Path to the audio file
y, sr = librosa.load(filename)

D = librosa.stft(y)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, y_axis='chroma_h', x_axis='time',
                               Sa=5, thaat='kafi', ax=ax)
ax.set(title='Chromagram with Hindustani notation')
fig.colorbar(img, ax=ax)

plt.show()