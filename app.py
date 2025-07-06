
from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    file.save(filename)

    # OpenCV - detecta rosto
    img = cv2.imread(filename)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    result = {}
    if len(faces) > 0:
        # Aqui entra o modelo IA de verdade
        result = {
            "Europeu": 70,
            "Indígena": 30
        }
    else:
        result = {
            "Erro": "Nenhum rosto detectado. Tente outra foto!"
        }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
