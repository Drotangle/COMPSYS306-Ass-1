import pandas as pd
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import datetime


def print_time(started=True, seconds=False):
    status = "Started"
    if not started:
        status = "Finished"

    current_time = datetime.datetime.now().time()
    if seconds:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}:{current_time.second}')
    else:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}')


def read_categories():
    labelsFile = "Assignment-Dataset/labels.csv"
    df = pd.read_csv(labelsFile)
    return df.to_numpy()


def read_in_data():
    # first have a look at the data
    # and put it into a dataframe

    # what do these mean?
    flat_data_arr = []
    target_arr = []

    categoryArray = read_categories()

    dataDirectory = 'Assignment-Dataset/myData'

    # add in all files
    Category_numbers = range(0, 43)

    for i in Category_numbers:

        # change it to category name
        cat_name = categoryArray[i][1]

        print(f'loading... category ({i}/42) :\t{cat_name}')
        path = os.path.join(dataDirectory, str(i))
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (32, 32, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(cat_name)
        print(f'loaded category : {cat_name} successfully')
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    print(df.shape)
    pickle.dump(df, open('data.pickle', 'wb'))


def open_data():
    return pickle.load(open('data.pickle', 'rb'))


# only call if its the first time
# read_in_data()

# we should probably label the data as well after this
# do we have to do anything to do the jpegs? - look at the extract py file
print_time(True, True)
df = open_data()
print_time(started=False, seconds=True)
