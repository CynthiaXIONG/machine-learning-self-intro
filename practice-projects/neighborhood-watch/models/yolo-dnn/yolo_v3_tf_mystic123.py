import os
import sys
import random

import colorsys
import imghdr
from PIL import Image, ImageDraw, ImageFont

import numpy as np

import tensorflow as tf


sys.path.append(os.path.join(os.path.dirname(__file__), './yolov3_mystic123_tensorflow'))
from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
from yolo_v3_tiny import yolo_v3_tiny

sys.path.append(os.path.dirname(__file__))
from model_utils import ObjectDetectionModel

FLAGS = tf.app.flags.FLAGS

class YoloV3Mystic123(ObjectDetectionModel): 
    name = "YoloV3Mystic123"

    def __init__(self, score_threshold=0.6, iou_threshold=0.5, use_tiny=False):
        #tiny model not working...

        classes_path = os.path.join(os.path.dirname(__file__), "yolov3_mystic123_tensorflow/coco.names")
        weights_path = os.path.join(os.path.dirname(__file__), "yolov3_mystic123_tensorflow/yolov3.weights")
        if (use_tiny):
            self.name = "YoloV3TinyMystic123"
            weights_path = os.path.join(os.path.dirname(__file__), "yolov3_mystic123_tensorflow/tiny.weights")
        
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        
        self.classes = self._read_classes(classes_path)

        # Config default boxes colors
        self.boxes_colors = self._generate_colors(self.classes)

        # placeholder for detector inputs
        self.model_input_size = 416
        self.inputs = tf.placeholder(tf.float32, [None, self.model_input_size, self.model_input_size, 3])

        with tf.variable_scope('detector'):
            if (use_tiny):
                self.detections = yolo_v3_tiny(self.inputs, len(self.classes), data_format='NCHW')
            else:
                self.detections = yolo_v3(self.inputs, len(self.classes), data_format='NCHW')
            load_ops = load_weights(tf.global_variables(scope='detector'), weights_path)

        self.boxes = detections_boxes(self.detections)

        #start the session
        self.tf_session = tf.Session()
        self.tf_session.run(load_ops)

    def predict(self, image_file_name):

        image = Image.open(self._get_input_filename(image_file_name))
        image_resized = image.resize(size=(self.model_input_size,self.model_input_size))

        detected_boxes = self.tf_session.run(self.boxes, feed_dict={self.inputs: [np.array(image_resized, dtype=np.float32)]})

        filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=self.score_threshold, iou_threshold=self.iou_threshold)

        self._draw_boxes(filtered_boxes, image, self.classes, (self.model_input_size, self.model_input_size), self.boxes_colors)

        # Save the predicted bounding box on the image
        image.save(self._get_output_filename(image_file_name), quality=90)

        #return result
        return self._build_result_output(filtered_boxes, self.classes);

    def _generate_colors(self, class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    def _build_result_output(self, boxes, cls_names):
        result = []
        for cls, bboxs in boxes.items():
            num_bboxs = len(bboxs)
            if (num_bboxs > 0):
                item = {}
                item["name"] = cls_names[cls]
                item["count"] = num_bboxs
                result.append(item)

        return result

    def _draw_boxes(self, boxes, image, cls_names, detection_size, colors):
        font_path = os.path.join(os.path.dirname(__file__), "font/consola.otf")
        font = ImageFont.truetype(font=font_path,size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        draw = ImageDraw.Draw(image)

        for cls, bboxs in boxes.items():
            for box, score in bboxs:
                box = self._convert_to_original_size(box, np.array(detection_size), np.array(image.size))
                label = '{} {:.2f}'.format(cls_names[cls], score*100)

                for i in range(thickness): #hacky way to draw thick rectangles
                    draw.rectangle((np.array(box) + i).tolist(), outline=colors[cls])

                text_origin = np.array(box[:2])
                label_size = np.array(draw.textsize(label, font))
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[cls])
                draw.text(text_origin.tolist(), label, fill=(0, 0, 0), font=font)

                """
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                
                for i in range(thickness): #hacky way to draw thick rectangles
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[cls])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[cls])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                """

    def _convert_to_original_size(self, box, size, original_size):
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))

    def _read_classes(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


#--------- MAIN ---------#
def main(argv=None):

    yolo_v3_model = YoloV3Mystic123()

    yolo_v3_model.predict("pi3")


if __name__ == "__main__":
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
    tf.app.run() #runs faster than just calling main in debug atleast
