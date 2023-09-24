import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score


def save_mlp_model(model):
    joblib.dump(model, 'mlp_model.pkl')


def load_mlp_model():
    return joblib.load('mlp_model.pkl')


def fit_and_train_mlp_model(x_training, x_valid, y_training, y_valid, learning_rate, iterations, save_model=False):
    # may want to make early stopping false (cos might have two validation sets?)
    model = MLPClassifier(hidden_layer_sizes=(3072, 1028, 256, 43), random_state=1, learning_rate_init=learning_rate,
                          max_iter=iterations, early_stopping=True)
    model.fit(x_training, y_training)

    # now validate it
    y_pred = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='macro')
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")

    # save the model to joblib file if we want to
    if save_model:
        save_mlp_model(model)