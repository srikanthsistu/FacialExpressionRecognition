import torch
import numpy as np
import os
import sys
import cv2
import glog as log
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt


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


def create_data_loader():
    transform = transforms.ToTensor() # TODO Add better transforms
    train_data = datasets.ImageFolder(os.path.join(os.getcwd(), "data/images/publictest"), transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    for idx, (images, targets) in enumerate(train_loader):
        if idx == 0:
            img = torchvision.utils.make_grid(images)
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.savefig("first_batch.png")


def main():
    out_path = os.path.join(os.getcwd(), "data/images")
    inp_path = os.path.join(os.getcwd(), "data/fer2013/fer2013.csv")
    log.info("Input path is {}".format(inp_path))
    if len(os.listdir(out_path)) == 0:
        convert_csv_to_image(out_path, inp_path)
    create_data_loader()


if __name__ == '__main__':
    main()
