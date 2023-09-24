import joblib
from sklearn import svm
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import cv2


def save_svm_model(model):
    joblib.dump(model, 'svm_model.pkl')


def load_svm_model():
    return joblib.load('svm_model.pkl')


def fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, save_model=False):

    # we only use hog cos some colors are really off, might not be useful

    # note the first value here are dependent on the splits
    # also, this is just so we can get HOG from the training set
    x_training.reshape(52660, 32, 32, 3)
    hog_features, hog_image = hog(x_training, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  block_norm='L2-Hys', visualize=True)

    svc = svm.SVC(kernel='rbf', gamma=0.7, C=3, max_iter=10000, probability=True)
    svc.fit(x_training, y_training)


def hog_test(x_training):
    # note: this below works!
    # x_big_set = x_training.reshape(52660, 32, 32, 3)
    # image_flat = x_big_set[30000]

    image_flat = x_training[20000, :]
    image = np.array(image_flat).reshape(32, 32, 3)

    # block size of 16x16 as that is usually the size of the sign
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
