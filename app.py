from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from utils import *
from flask import send_file
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'static/temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


path = "./static/images/uploaded_image.jpg"

scale = 3
hp = 210 * scale
wp = 297 * scale

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")

@app.route('/')
def index():
    return render_template('index.html')

def detect_dimensions(frame):
    print("Detecting dimensions...")
    imgContour, fc = getContours(frame, minArea=50000, filter=4, draw=True)

    if len(fc) != 0:
        biggestCont = fc[0][2]
        cv2.drawContours(imgContour, biggestCont, -1, (0, 0, 255), 3)
        imgWarp = warpImg(frame, biggestCont, wp, hp)

        imgContour2, fc2 = getContours(imgWarp, minArea=2000, filter=4, threshold=[50, 50], draw=False)

        if len(fc2) != 0:
            for obj in fc2:
                cv2.polylines(imgContour2, [obj[2]], True, (0, 255, 0), 2)
                npoints = reorder(obj[2])
                newW = round((findDis(npoints[0][0] // scale, npoints[1][0] // scale) / 10), 1)
                newH = round((findDis(npoints[0][0] // scale, npoints[2][0] // scale) / 10), 1)
                cv2.arrowedLine(imgContour2, (npoints[0][0][0], npoints[0][0][1]),
                                (npoints[1][0][0], npoints[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContour2, (npoints[0][0][0], npoints[0][0][1]),
                                (npoints[2][0][0], npoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContour2, '{}cm'.format(newW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContour2, '{}cm'.format(newH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', imgContour2)
            return buffer.tobytes()

    # If no contours are detected, return an empty byte string
    print("No contours detected...")
    return b''


@app.route('/upload', methods=['POST'])
def upload_image():
    print("Here - upload_image")

    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        file.save(path)
        img = cv2.imread(path)
        print("Image loaded")
        frame_bytes = detect_dimensions(img)

        print(f"Length of frame_bytes: {len(frame_bytes)}")

        if frame_bytes:
            print("Dimensions detected")
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
            cv2.imwrite(temp_path, cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR))
            # return render_template('index.html', image_path=path, image_bytes=frame_bytes)
            return send_file(temp_path, mimetype='image/jpeg')
        else:
            print("No dimensions detected")
            return render_template('error.html', error="No dimensions detected")

    return render_template('error.html', error="Unexpected error")



if __name__ == '__main__':
    app.run(debug=True)
