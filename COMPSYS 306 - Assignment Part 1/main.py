# first have a look at the data
# and put it into a dataframe
import pandas as pd

autodataset = '/content/gdrive/MyDrive/CS306_2023/database/numerical/auto-mpg.csv'
datafr = pd.read_csv(autodataset)

print(datafr.shape)
print(datafr.size)