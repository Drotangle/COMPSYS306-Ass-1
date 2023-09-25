import joblib
from sklearn import svm
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import cv2

import show_time


def save_svm_model(model):
    joblib.dump(model, 'svm_model.joblib')


def load_svm_model():
    return joblib.load('svm_model.joblib')


def fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, save_model=False):
    # we only use hog cos some colors are really off, might not be useful
    hog_features_training = []
    hog_features_valid = []

    # apply hog on the data to get features
    x_training_not_flat = x_training.reshape(52660, 32, 32, 3)
    for image in x_training_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_training.append(hog_features)

    x_valid_not_flat = x_valid.reshape(13165, 32, 32, 3)
    for image in x_valid_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_valid.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # train the model
    model = svm.SVC(kernel='rbf', gamma=0.7, C=3, max_iter=1400, probability=True)
    model.fit(hog_features_training, y_training)

    # do validation on the current params
    y_pred = model.predict(hog_features_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='micro')
    score = f1_score(y_valid, y_pred, average="micro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"F1 Score: {score * 100}%")

    if save_model:
        save_svm_model(model)


def validation(x_testing, y_testing):
    print("\nSVM testing info:\n")

    # check model based off of testing params
    model = load_svm_model()

    # hog the model
    hog_features_testing = []

    x_testing_not_flat = x_testing.reshape(7314, 32, 32, 3)
    for image in x_testing_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_testing.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # test it by predicting
    y_pred = model.predict(hog_features_testing)
    accuracy = accuracy_score(y_testing, y_pred)
    precision = precision_score(y_testing, y_pred, average='macro')
    recall = recall_score(y_testing, y_pred, average='macro')
    score = f1_score(y_true=y_testing, y_pred=y_pred, average="macro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"Recall Score: {recall * 100}%")
    print(f"F1 Score: {score * 100}%")
    print(classification_report(y_true=y_testing, y_pred=y_pred))
    print(confusion_matrix(y_testing, y_pred))


def individual_test(x_testing, y_testing):
    model = load_svm_model()

    # show the guess and actual for an image, to check if we are guessing correctly
    img_num = 1

    image_flat = x_testing[img_num, :]
    image = np.array(image_flat).reshape(32, 32, 3)
    hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), channel_axis=-1)

    print(f"prediction: {model.predict(np.array(hog_features).reshape(1,-1))[0]}")
    print(f"actual: {y_testing[img_num]}")