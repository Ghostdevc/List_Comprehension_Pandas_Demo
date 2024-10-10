import seaborn as sns
import numpy as np
import pandas as pd

df = sns.load_dataset('car_crashes')
print(df.columns)
columns = df.columns
columns = [('NUM_' + col) if df[df.columns].dtypes != "O" else col for col in [col.upper() for col in columns]]
columns

columns = df.columns
columns = [(col + '_FLAG').upper() if 'no'  not in col else col.upper() for col in columns]
columns

columns = df.columns
og_list = ['abbrev', 'no_previous']
new_cols = [col for col in columns if col not in og_list]
new_cols
new_df = df[new_cols]



titanic_df = sns.load_dataset('titanic')
titanic_df.head()

titanic_df.groupby('sex').count()

