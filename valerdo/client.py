import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "test\\jimmy hendrix-Purple Haze.wav"
NUM_SAMPLES_TO_CONSIDER = 22050 * 3

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    # halfway_point = len(sound) / 2
    # audio_file = sound[halfway_point - NUM_SAMPLES_TO_CONSIDER:halfway_point + NUM_SAMPLES_TO_CONSIDER]

    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted genre is: {data['genre']}")
