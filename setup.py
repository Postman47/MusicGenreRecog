import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                         Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random

# os.makedirs('C:/Users/Piotrek/Desktop/PW/Praca_inz/MusicGenreRecog/content/spectograms3sec')
# os.makedirs('C:/Users/Piotrek/Desktop/PW/Praca_inz/MusicGenreRecog/content/spectograms3sec/train')
# os.makedirs('C:/Users/Piotrek/Desktop/PW/Praca_inz/MusicGenreRecog/content/spectograms3sec/test')

genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()
for g in genres:
    path_audio = os.path.join('E:/Praca_inz/MusicGenreRecog/content/audio3sec', f'{g}')
    os.makedirs(path_audio)
    path_train = os.path.join('E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/train', f'{g}')
    path_test = os.path.join('E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/test', f'{g}')
    os.makedirs(path_train)
    os.makedirs(path_test)