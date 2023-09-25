import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def save_mlp_model(model):
    joblib.dump(model, 'mlp_model.pkl')


def load_mlp_model():
    return joblib.load('mlp_model.pkl')


def fit_and_train_mlp_model(x_training, x_valid, y_training, y_valid, learning_rate, iterations, save_model=False):

    # may want to make early stopping false (cos might have two validation sets?)
    # first 2 layers are hidden, 3rd is output it seems (or not?)
    model = MLPClassifier(hidden_layer_sizes=(1024, 512), random_state=1, learning_rate_init=learning_rate,
                          max_iter=iterations, early_stopping=True)
    model.fit(x_training, y_training)

    # now validate it
    y_pred = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='macro')
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(classification_report(y_true=y_valid, y_pred=y_pred))

    # save the model to joblib file if we want to
    if save_model:
        save_mlp_model(model)


def validation(x_testing, y_testing):

    # check model based off of testing params
    model = load_mlp_model()
    y_pred = model.predict(x_testing)
    print(classification_report(y_true=y_testing, y_pred=y_pred))