# note: to install more packages: do .\venv\Scripts\Activate first!
import pandas as pd
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt

import mlp_model
import svm_model


def print_time(started=True, seconds=False):
    status = "Started"
    if not started:
        status = "Finished"

    current_time = datetime.datetime.now().time()
    if seconds:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}:{current_time.second}')
    else:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}')


def plot_category_sizes(category_sizes):
    bar_x_values = range(0,43)
    plt.bar(bar_x_values, category_sizes)
    plt.xlabel("category #")
    plt.ylabel("category frequency")
    plt.show()

def read_categories():
    labelsFile = "Assignment-Dataset/labels.csv"
    df = pd.read_csv(labelsFile)
    return df.to_numpy()


def read_in_data(save_to_file = True):
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


def split_dataset(df, test_size):

    x = df.drop(['Target'], axis = 1).values
    y = df['Target'].values

    return train_test_split(x,y,test_size = test_size)


# we should probably label the data as well after this
# do we have to do anything to do the jpegs? - look at the extract py file
print_time(True, True)

# get data from file
df, category_sizes = open_data()
quick_analysis(df)

print_time(started=False, seconds=True)

# plot bar graph of size of each category
plot_category_sizes(category_sizes)

# split into training and testing datasets
x_training,x_testing,y_training,y_testing = split_dataset(df, 0.2)