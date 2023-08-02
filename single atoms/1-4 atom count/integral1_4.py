import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for i in range(1, 40):
        img = cv2.imread(os.path.join(folder, 'atom {}.tif'.format(i)), 0)
        if img is not None:
            images.append(img)
        else:
            print(i)
    return images


folder = '/Users/carrie/PycharmProjects/cluster/single atoms/1-4 atom count/1-4 atom count'
images = load_images_from_folder(folder)
avg = []
for img in images:
    bg = img < 50
    img[bg] = 0

    inte = np.average(img)
    avg.append(inte)
