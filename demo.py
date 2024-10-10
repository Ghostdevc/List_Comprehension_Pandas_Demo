import seaborn as sns
import numpy as np
import pandas as pd

df = sns.load_dataset('car_crashes')
print(df.columns)
[('NUM_' + col).lower() if df[col].dtypes != "O" else col.upper() for col in df.columns]

[(col + '_FLAG').upper() if 'no'  not in col else col.upper() for col in df.columns]

og_list = ['abbrev', 'no_previous']
new_cols = [col for col in df.columns if col not in og_list]
new_cols
new_df = df[new_cols]



titanic_df = sns.load_dataset('titanic')
titanic_df.head()

titanic_df['sex'].value_counts()

titanic_df.nunique()

titanic_df['pclass'].nunique()

titanic_df.loc[:, ['pclass', 'parch']].nunique()

titanic_df['embarked'].dtype
titanic_df['embarked'] = titanic_df['embarked'].astype('category')
titanic_df.dtypes

titanic_df[titanic_df['embarked'] == 'C']

titanic_df[titanic_df['embarked'] != 'S']

titanic_df.loc[(titanic_df['age'] < 30) & (titanic_df['sex'] == 'female')]

titanic_df.loc[(titanic_df['fare'] > 500) | (titanic_df['age'] > 70)]

titanic_df.isna().sum()

titanic_df.drop('who', axis=1, inplace= True)

titanic_df['deck'].fillna(titanic_df['deck'].mode(), inplace=True)

titanic_df['age'].fillna(titanic_df['age'].median(), inplace=True)

titanic_df.groupby(['pclass', 'sex']).agg({'survived' : ['sum', 'count', 'mean']})

titanic_df['age_flag'] = titanic_df['age'].apply(lambda x: 1 if x < 30 else 0)
titanic_df['age_flag'].head()




tips_df = sns.load_dataset('tips')

tips_df.groupby('time').agg({'total_bill': ['sum', 'min', 'max', 'mean']})

tips_df.groupby(['day', 'time']).agg({'total_bill': ['sum', 'min', 'max', 'mean']})

tips_df[(tips_df['sex'] == 'female')  & (tips_df['time'] == 'Lunch')].groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean']})

tips_df.loc[(tips_df['size'] < 3) & (10 < tips_df['total_bill'])].agg({'total_bill': 'mean'})

tips_df['total_bill_tip_sum'] = tips_df['total_bill'] + tips_df['tip']
tips_df['total_bill_tip_sum'].head()

new_df = tips_df.sort_values(by='total_bill_tip_sum', ascending=False)[0:30]
new_df.head()