import os
import librosa
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd


genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()

for g in genres:
    for filename in os.listdir(os.path.join('E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/all',f"{g}")):
        path = os.path.join(f'E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/all/{g}', f'{filename}')

        image = Image.open(path)
        a = librosa.db_to_power(image)
        X = librosa.feature.mfcc(a)
        print(X)
