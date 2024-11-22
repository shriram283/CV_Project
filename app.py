from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os

app = Flask(__name__)

def load_image(path, scale=0.7):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (0, 0), None, scale, scale)
    return img_resized

def preprocess_image(img, thresh_1=57, thresh_2=232):
    img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur  = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, thresh_1, thresh_2)

    kernel = np.ones((3, 3))
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)
    img_closed = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel, iterations=4)

    return img_closed

def find_contours(img_preprocessed, img_original, epsilon_param=0.04):
    contours, _ = cv2.findContours(image=img_preprocessed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    img_contour = img_original.copy()
    cv2.drawContours(img_contour, contours, -1, (203, 192, 255), 6)
    
    print("Number of contours found:", len(contours))  
    
    polygons = []
    for contour in contours:
        epsilon = epsilon_param * cv2.arcLength(curve=contour, closed=True)
        polygon = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
        if len(polygon) == 4:
            polygons.append(polygon.reshape(4, 2))
    
    return polygons, img_contour

def reorder_coords(polygon):
    rect_coords = np.zeros((4, 2))

    add = polygon.sum(axis=1)
    rect_coords[0] = polygon[np.argmin(add)]
    rect_coords[3] = polygon[np.argmax(add)]

    subtract = np.diff(polygon, axis=1)
    rect_coords[1] = polygon[np.argmin(subtract)]
    rect_coords[2] = polygon[np.argmax(subtract)]

    return rect_coords

def warp_image(rect_coords, img_original, pad=5):
    rect_coords = np.array(rect_coords, dtype=np.float32)

    # Define the output size based on the original image dimensions (no A4 reference)
    img_h, img_w = img_original.shape[:2]
    output_size = (img_w, img_h)

    matrix = cv2.getPerspectiveTransform(src=rect_coords, dst=np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]]))
    img_warped = cv2.warpPerspective(img_original, matrix, output_size)
    
    warped_h, warped_w = img_warped.shape[:2]
    img_warped = img_warped[pad:warped_h-pad, pad:warped_w-pad]
    
    return img_warped

def calculate_sizes(polygons_warped, dpi=96):
    sizes = []
    for polygon in polygons_warped:
        height_px = cv2.norm(polygon[0], polygon[2], cv2.NORM_L2)
        width_px  = cv2.norm(polygon[0], polygon[1], cv2.NORM_L2)
        
        height_cm = (height_px / dpi) * 2.54
        width_cm  = (width_px / dpi) * 2.54
        
        sizes.append([height_cm, width_cm])
    return np.array(sizes)

def measure_size(path):
    img_original = load_image(path)
    img_preprocessed = preprocess_image(img_original)
    polygons, img_contours = find_contours(img_preprocessed, img_original)
    
    if not polygons:
        return None, "No valid contours detected"

    # Get dimensions
    dimensions = img_original.shape[:2]  
    rect_coords = reorder_coords(polygons[0])

    img_warped = warp_image(rect_coords, img_original)
    img_warped_preprocessed = preprocess_image(img_warped)
    polygons_warped, img_contours_warped = find_contours(img_warped_preprocessed, img_warped)
    
    sizes = calculate_sizes(polygons_warped)

    # Prepare images to return
    return {
        "original": img_original,
        "preprocessed": img_preprocessed,
        "contours": img_contours,
        "warped": img_warped,
        "warped_preprocessed": img_warped_preprocessed,
        "contours_warped": img_contours_warped,
    }, sizes, dimensions

import base64

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = f"./static/{file.filename}"
            file.save(path)

            images, sizes, dimensions = measure_size(path)

            if images is not None:
                encoded_images = {}
                for key, img in images.items():
                    _, img_encoded = cv2.imencode('.png', img)
                    img_b64 = base64.b64encode(img_encoded).decode('utf-8')
                    encoded_images[key] = img_b64

                return render_template('result.html', images=encoded_images, sizes=sizes, dimensions=dimensions)

            else:
                return f"<h1>Error: {sizes}</h1>"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
