import joblib
from sklearn import svm
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, classification_report
import cv2

import show_time


def save_svm_model(model):
    joblib.dump(model, 'svm_model.pkl')


def load_svm_model():
    return joblib.load('svm_model.pkl')


def fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, save_model=False):
    # we only use hog cos some colors are really off, might not be useful
    hog_features_training = []
    hog_features_valid = []

    # note the first value here are dependent on the splits
    # also, this is just so we can get HOG from the training set
    x_training_not_flat = x_training.reshape(52660, 32, 32, 3)
    for image in x_training_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys', channel_axis=-1)
        hog_features_training.append(hog_features)

    x_valid_not_flat = x_valid.reshape(13165, 32, 32, 3)
    for image in x_valid_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys', channel_axis=-1)
        hog_features_valid.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    model = svm.SVC(kernel='rbf', gamma=0.7, C=3, max_iter=1400, probability=True)
    model.fit(hog_features_training, y_training)

    y_pred = model.predict(hog_features_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='macro')
    score = f1_score(y_valid, y_pred, average="macro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"F1 Score: {score * 100}%")

    if save_model:
        save_svm_model(model)


def validation(x_testing, y_testing):
    # check model based off of testing params
    model = load_svm_model()

    # hog the model
    hog_features_testing = []

    x_testing_not_flat = x_testing.reshape(7314, 32, 32, 3)
    for image in x_testing_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys', channel_axis=-1)
        hog_features_testing.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # test it
    y_pred = model.predict(hog_features_testing)
    accuracy = accuracy_score(y_testing, y_pred)
    precision = precision_score(y_testing, y_pred, average='macro')
    score = f1_score(y_testing, y_pred, average="macro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"F1 Score: {score * 100}%")

    print(classification_report(y_pred, y_testing))


def individual_test(x_testing, y_testing):
    model = load_svm_model()

    img_num = 1

    image_flat = x_testing[img_num, :]
    image = np.array(image_flat).reshape(32, 32, 3)
    hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys', channel_axis=-1)

    print(f"prediction: {model.predict(np.array(hog_features).reshape(1,-1))[0]}")
    print(f"actual: {y_testing[img_num]}")


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

# maybe see how much SIFT takes to do and try it?
