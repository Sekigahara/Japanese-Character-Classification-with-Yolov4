from flask import Flask, request, Response
import argparse
from darkeras_yolov4.core_yolov4 import utils

import models
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

@app.route("/api/detect", methods=["POSt"])
def detect():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        # Load image in bytes
        image_file = request.files["image"]
        image_bytes = image_file.read()

        # Load with opencv
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = cv2.resize(original_image, (INPUT_SIZE, INPUT_SIZE))
        image_data = image_data/255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        pred_bbox_array = np.array(pred_bbox)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, INPUT_SIZE, IOU_THRESH)
        bboxes = utils.nms(bboxes, IOU_THRESH, method='nms')

        image = models.draw_boxes(original_image, bboxes, classes_path=".././weight/obj.names")

        retval, buffer = cv2.imencode('.bmp', image)
        _txt_bmp = base64.b64encode(buffer)

        response = {'ImageBytes':_txt_bmp}
        response = pd.DataFrame.from_dict(response, orient='index')
        print(response)
        
        return Response(response.to_json(orient="records"), mimetype='application/json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API yolov4 for Japanese Character Recognition")
    parser.add_argument("--port", default=5000, type=int, help="port number to start")
    args = parser.parse_args()

    INPUT_SIZE = 736
    IOU_THRESH = 0.001

    # Load Model
    model = models.create_model(input_size=INPUT_SIZE, NUM_CLASS=46)
    # Load Weight
    utils.load_weights(model, ".././weight/yolov4-obj_12000_re.weights")
    utils.read_class_names(".././weight/obj.names")

    # Run Api
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port,debug=True)