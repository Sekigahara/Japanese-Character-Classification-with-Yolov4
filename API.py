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

app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect():
    # Load image in bytes
    image_file = request.args.get('image')
    image_bytes = base64.b64decode(image_file)
    img = Image.open(io.BytesIO(image_bytes))

    # Load with opencv
    original_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    original_image = cv2.rotate(original_image, cv2.cv2.ROTATE_90_CLOCKWISE)
    original_image_size = original_image.shape[:2]

    image_data = cv2.resize(original_image, (INPUT_SIZE, INPUT_SIZE))
    cv2.imwrite("test_saved/test.png", image_data)
    image_data = image_data/255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, INPUT_SIZE, IOU_THRESH)
    bboxes = utils.nms(bboxes, IOU_THRESH, method='nms')

    if bboxes == []:
        img = models.to_byte(img.to_list())
        response = {'Main_Image':img}
    else:
        image, cropped_image, predicted = models.draw_boxes(original_image, bboxes, classes_path="weight/obj.names")

        cv2.imwrite("test_saved/test2.png", np.array(image))
        # Main Image to Bytes
        main_img_bmp = models.to_byte(image)

        cropped_img_bmp = []
        # Sub cropped image to bytes
        for img_p in cropped_image:
            cropped_img_bmp.append(models.to_byte(np.float32(img_p)))
        cropped_img_bmp = np.asarray(cropped_img_bmp)

        #response = {'Main_Image':main_img_bmp, 'Cropped_Image':cropped_img_bmp, 'Predicted':predicted}
        response = {'Main_Image':main_img_bmp}

    json_ = json.dumps(response, ensure_ascii=False).encode('utf8')

    return Response(json_, mimetype='application/json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API yolov4 for Japanese Character Recognition")
    parser.add_argument("--port", default=5000, type=int, help="port number to start")
    args = parser.parse_args()

    INPUT_SIZE = 736
    IOU_THRESH = 0.001

    # Load Model
    model = models.create_model(input_size=INPUT_SIZE, NUM_CLASS=92)
    # Load Weight
    utils.load_weights(model, "weight/yolov4-obj_20100_re.weights")
    utils.read_class_names("weight/obj.names")

    # Run Api
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)