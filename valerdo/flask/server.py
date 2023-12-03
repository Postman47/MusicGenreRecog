from flask import Flask, request, jsonify
import random
from MGR_service import MGR_Service
import os

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke the MGR_Service
    mgrs = MGR_Service()

    # make a prediction
    predicted_genre = mgrs.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted keyword in json format
    data = {"genre": predicted_genre}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
