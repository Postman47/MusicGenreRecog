import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_template import FigureCanvas

genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()

for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.path.join('E:/Praca_inz/MusicGenreRecog/content/audio3sec',f"{g}")):
        song = os.path.join(f'E:/Praca_inz/MusicGenreRecog/content/audio3sec/{g}', f'{filename}')
        j = j+1

        y,sr = librosa.load(song, duration=3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,np.max))
        plt.savefig(f'E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/train/{g}/{g+str(j)}.png')
