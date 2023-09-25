import pandas as pd
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt

import mlp_model
import svm_model
import show_time


def plot_category_sizes(category_sizes):
    bar_x_values = range(0, 43)
    plt.bar(bar_x_values, category_sizes)
    plt.xlabel("category #")
    plt.ylabel("category frequency")
    plt.show()


def read_categories():
    labelsFile = "Assignment-Dataset/labels.csv"
    df = pd.read_csv(labelsFile)
    return df.to_numpy()


def read_in_data(save_to_file=True):
    # first have a look at the data
    # and put it into a dataframe

    # initialize array that hold flattened, normalized image data
    # as well as array that holds target info to append to the above one
    flat_data_arr = []
    target_arr = []

    # this is for data analysis of the dataset
    category_sizes = []

    categoryArray = read_categories()

    dataDirectory = 'Assignment-Dataset/myData'

    # add in all files
    Category_numbers = range(0, 43)

    for i in Category_numbers:

        # change it to category name
        cat_name = categoryArray[i][1]

        # initialize count of images in a category
        category_count = 0

        print(f'loading... category ({i}/42) :\t{cat_name}')
        path = os.path.join(dataDirectory, str(i))

        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            # resizing is important to ensure all images are 32 by 32 px
            # also, this normalizes the data for us
            img_resized = resize(img_array, (32, 32, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(i)
            category_count += 1

        category_sizes.append(category_count)
        print(f'loaded category : {cat_name} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    print(df.shape)

    # only if we want to save to file, do this step
    if save_to_file:
        pickle.dump(df, open('data.pickle', 'wb'))
        pickle.dump(category_sizes, open('category_sizes.pickle', 'wb'))


def open_data():
    return pickle.load(open('data.pickle', 'rb')), pickle.load(open('category_sizes.pickle', 'rb'))


def quick_analysis(dataframe):
    print("\nData Analysis:")
    print(f'shape: {dataframe.shape}')
    print(f"first 10 :\n{dataframe.head(10)}")
    offset = 70000
    first_10_with_offset = dataframe.iloc[offset:offset + 10]
    print(f"random 10 :\n{first_10_with_offset}")
    print(f"")


def split_dataset(df_split, valid_size, test_size, do_print=False, do_save_file=False):
    x = df_split.drop(['Target'], axis=1).values
    y = df_split['Target'].values

    # note that the validate size is based off the size of the validate and training sets
    x_train_val, x_testing_split, y_train_val, y_testing_split = train_test_split(x, y,
                                                                                  test_size=test_size,
                                                                                  random_state=1)
    x_training_split, x_valid_split, y_training_split, y_valid_split = train_test_split(x_train_val, y_train_val,
                                                                                        test_size=valid_size,
                                                                                        random_state=1)

    # print out if we want to see the shape of our datasets for training / testing / validation
    if do_print:
        print(np.shape(x_training_split))
        print(np.shape(x_testing_split))
        print(np.shape(x_valid_split))
        print(np.shape(y_training_split))
        print(np.shape(y_testing_split))
        print(np.shape(y_valid_split))

    if do_save_file:
        joblib.dump(x_training_split, "x_training.pkl")
        joblib.dump(x_testing_split, "x_testing.pkl")
        joblib.dump(x_valid_split, "x_valid.pkl")
        joblib.dump(y_training_split, "y_training.pkl")
        joblib.dump(y_testing_split, "y_testing.pkl")
        joblib.dump(y_valid_split, "y_valid.pkl")

    return x_training_split, x_testing_split, x_valid_split, y_training_split, y_testing_split, y_valid_split


def load_split_dataset():
    x_training_split = joblib.load("x_training.pkl")
    x_testing_split = joblib.load("x_testing.pkl")
    x_valid_split = joblib.load("x_valid.pkl")
    y_training_split = joblib.load("y_training.pkl")
    y_testing_split = joblib.load("y_testing.pkl")
    y_valid_split = joblib.load("y_valid.pkl")

    return x_training_split, x_testing_split, x_valid_split, y_training_split, y_testing_split, y_valid_split


# we should probably label the data as well after this
# do we have to do anything to do the jpegs? - look at the extract py file
show_time.print_time(True, True)

# get data from file
# df, category_sizes = open_data()
# quick_analysis(df)

# look for median of bar graph
'''median = sorted(category_sizes)[21]
new_category_sizes = []
print(median)
for value in category_sizes:
    if value < median:
        new_category_sizes.append(value)
    elif value > median:
        # maybe we could just third these categories and that's it?
        new_category_sizes.append(value / 4)
    else:
        new_category_sizes.append(value)

# plot bar graph of size of each category
plot_category_sizes(new_category_sizes)'''

# split into training and testing and validation datasets
# split_dataset(df, 0.2, 0.1, True, True)
x_training, x_testing, x_valid, y_training, y_testing, y_valid = load_split_dataset()

# try train the model
mlp_model.fit_and_train_mlp_model(x_training, x_valid, y_training, y_valid, 0.01, 2500, True)
# svm_model.fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, True)

# svm_model.hog_test(x_training)

# now that we have correct hyperparameters, using testing dataset
# svm_model.individual_test(x_testing, y_testing)
# mlp_model.validation(x_testing, y_testing)

show_time.print_time(False, True)
