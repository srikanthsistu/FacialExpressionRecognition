import torch
import numpy as np
import os
import sys
import cv2
import glog as log


def convert_csv_to_image(output_path, input_path=""):
    if os.path.exists(output_path):
        os.system('rm -rf {}'.format(output_path))

    os.system('mkdir {}'.format(output_path))
    label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    data = np.genfromtxt(input_path, delimiter=',', dtype=None, encoding=None)
    labels = data[1:, 0].astype(np.int32)
    image_buffer = data[1:, 1]
    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
    usage = data[1:, 2]
    dataset = zip(labels, images, usage)
    for i, d in enumerate(dataset):
        if d[-1] == "Training" or d[-1] == "PrivateTest":
            usage_path = os.path.join(output_path, "train")
        else:
            usage_path = os.path.join(output_path, d[-1].lower())
        if not os.path.exists(usage_path):
            os.mkdir(usage_path)

        label_path = os.path.join(usage_path, label_names[d[0]])
        if not os.path.exists(label_path):
            os.mkdir(label_path)

        img = d[1].reshape((48, 48))
        img_name = '%08d.jpg' % i
        img_path = os.path.join(label_path, img_name)
        if not os.path.exists(usage_path):
            os.system('mkdir {}'.format(usage_path))
        if not os.path.exists(label_path):
            os.system('mkdir {}'.format(label_path))
        cv2.imwrite(img_path, img)


out_path = os.path.join(os.getcwd(), "data/images")
inp_path = os.path.join(os.getcwd(), "data/fer2013/fer2013.csv")
log.info("Input path is {}".format(inp_path))

convert_csv_to_image(out_path, inp_path)
