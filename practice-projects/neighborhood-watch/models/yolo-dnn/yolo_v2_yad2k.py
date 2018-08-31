#yad2k - the keras/tensorflow implementation of YOLO_V2/3
import os
import random

import colorsys
import imghdr
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from keras import backend as K
from keras.models import load_model
from yad2k.yad2k.models.keras_yolo import yolo_eval, yolo_head

from model_utils import ObjectDetectionModel

class YoloV2Yad2k(ObjectDetectionModel): 
    name = "YoloV2Yad2k"

    def __init__(self, model_path="yad2k/model_data/yolov2.h5", \
    anchors_path="yad2k/model_data/yolo_anchors.txt", \
    classes_path="yad2k/model_data/coco_classes.txt",\
    score_threshold=0.6, iou_threshold=0.5):
        self.sess = K.get_session()

        # use ./yad2k.py model_data/yolo.cfg model_data/yolo.weights model_data/yolo.h5 to convert .cfg/.weights to .h5 (keras)

        self.yolo_model = load_model(model_path) 
        self.class_names = self._read_classes(classes_path)
        self.anchors = self._read_anchors(anchors_path)

        # Generate colors for drawing bounding boxes.
        self.bb_colors = self._generate_colors(self.class_names)

        # Verify model, anchors, and classes are compatible
        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)

        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        #self.is_fixed_size = self.model_image_size != (None, None)

        # Config default boxes colors
        self.boxes_colors = self._generate_colors(self.class_names)

        # Generate output tensor targets for filtered bounding boxes.
        # Convert output of the model to usable bounding box tensors
        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo_outputs, self.input_image_shape, score_threshold=score_threshold, iou_threshold=iou_threshold)

    def predict(self, image_file_name):
        # Preprocess your image
        image, image_data = self.preprocess_image(self._get_input_filename(image_file_name), model_image_size=self.model_image_size)

        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        out_scores, out_boxes, out_classes = self.sess.run([self.scores, self.boxes, self.classes], feed_dict={self.yolo_model.input:image_data, \
                                                                                                               self.input_image_shape: [image.size[1], image.size[0]], \
                                                                                                               K.learning_phase(): 0})

        # Print predictions info
        print('Found {} boxes for {}'.format(len(out_boxes), image_file_name))

        # Draw bounding boxes on the image file
        self._draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, self.boxes_colors)
        # Save the predicted bounding box on the image
        image.save(self._get_output_filename(image_file_name), quality=90)

        return out_scores, out_boxes, out_classes


    def preprocess_image(self, img_path, model_image_size):
        image_type = imghdr.what(img_path)
        image = Image.open(img_path)
        resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image, image_data

    def _read_classes(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _read_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _generate_colors(self, class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    def _draw_boxes(self, image, out_scores, out_boxes, out_classes, class_names, colors):
        font = ImageFont.truetype(font='font/consola.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            del draw
    

def main():
    # Change dir to this script location
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force to run on CPU (GPU-tf not working properly)

    yolo_v2 = YoloV2Yad2k()
    yolo_v2.predict("pi3")
    

if __name__ == "__main__":
    main()
