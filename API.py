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
import json
import io
from PIL import Image
from timeit import default_timer as timer
from datetime import timedelta

app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect():
    # Load image in bytes
    image_json = request.get_json(force=True)
    image_file = image_json["Main_Image"]
    image_bytes = base64.b64decode(image_file)
    img = Image.open(io.BytesIO(image_bytes))

    # Load with opencv
    original_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    #original_image = cv2.rotate(original_image, cv2.cv2.ROTATE_90_CLOCKWISE)
    original_image_size = original_image.shape[:2]

    #original_image  = cv2.imread("test_saved/tryraw1.png")
    image_data = cv2.resize(original_image, (INPUT_SIZE, INPUT_SIZE))
    image_data = image_data/255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    start = timer()
    pred_bbox = model.predict(image_data)
    end = timer()
    print("Elapsed Time : ", timedelta(seconds=end-start))
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, [736, 736], INPUT_SIZE, IOU_THRESH)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    if bboxes == []:
        crop = np.resize(original_image, (256, 256))
        crop = models.to_byte(crop)
        img = models.to_byte(original_image)
        print("None Detected")
        response = {'Main_Image':img, 'Cropped_Image':[crop], 'Predicted':['No Detection']}
    else:
        image, cropped_image, predicted = models.draw_boxes(original_image, bboxes, classes_path="weight/obj1.names", show_label=True)

        # Main Image to Bytes
        main_img_bmp = models.to_byte(image)

        cropped_img_bmp = []
        # Sub cropped image to bytes
        for img_p in cropped_image:
            cropped_img_bmp.append(models.to_byte(np.float32(img_p)))

        print("Detected")
        response = {'Main_Image':main_img_bmp, 'Cropped_Image':cropped_img_bmp, 'Predicted':predicted}
        #response = {'Main_Image':main_img_bmp}

    json_ = json.dumps(response, ensure_ascii=False).encode('utf8')

    return Response(json_, mimetype='application/json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API yolov4 for Japanese Character Recognition")
    parser.add_argument("--port", default=5000, type=int, help="port number to start")
    args = parser.parse_args()

    INPUT_SIZE = 736
    IOU_THRESH = 0.005

    # Load Model
    model = models.create_model(input_size=INPUT_SIZE, NUM_CLASS=46)
    # Load Weight
    utils.load_weights(model, "weight/yolov4-obj_22300_hira.weights")
    utils.read_class_names("weight/obj1.names")

    # Run Api
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=False)