from flask import Flask, request, send_from_directory, jsonify
import os
import cv2
from prediction import predict_image  # Import the function from prediction.py

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Get the file from POST request

    # Save the file to UPLOAD_FOLDER
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Convert the image file to a numpy array
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Predict the colorized image
    result = predict_image(img)

    # Save the result to OUTPUT_FOLDER
    output_filename = "color_" + filename
    output_filepath = os.path.join(
        app.config['OUTPUT_FOLDER'], output_filename)
    cv2.imwrite(output_filepath, result)

    # Send a response with the filename of the colorized image
    return jsonify({"image": output_filename})


@app.route('/output/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
