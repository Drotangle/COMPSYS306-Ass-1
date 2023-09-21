# first have a look at the data
# and put it into a dataframe
import pandas as pd

dataDirectory = 'Assignment-Dataset/myData'
dataframe = pd.read_csv(dataDirectory)

print(dataframe.shape)
print(dataframe.size)