import librosa
import numpy as np
import tensorflow.keras as keras
from keras_visualizer import visualizer
from pydub import AudioSegment
AudioSegment.converter = "C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ffmpeg\\ffmpeg\\bin\\ffprobe.exe"

MODEL_PATH = "model_cnn_512_512_128_lr=0001_332_v3_biasall"
NUM_SAMPLES_TO_CONSIDER = 22050 * 3

class _MGR_Service:

    model = None
    _mappings = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]
    _instance = None


    def predict(self, file_path):

        # etract the  MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficients)

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_genre = self._mappings[predicted_index]

        return predicted_genre

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)
        start = 0

        # ensure consistency in the audio length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            start = int((len(signal)-NUM_SAMPLES_TO_CONSIDER)/2)

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal[start:start+NUM_SAMPLES_TO_CONSIDER], sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

    def Visualize(self):

        visualizer(keras.models.load_model(MODEL_PATH), format='png', view=True)

def MGR_Service():

    # ensure that we only have one instance of MGR_Service
    if _MGR_Service._instance is None:
        _MGR_Service._instance = _MGR_Service()
        _MGR_Service.model = keras.models.load_model(MODEL_PATH)
    return _MGR_Service._instance

if __name__ == "__main__":

    MGRS = MGR_Service()

    MGRS.Visualize()

    genre1 = MGRS.predict("test\\Queen - I Want To Break Free.wav")

    print(f"Predicted keywords: {genre1}")