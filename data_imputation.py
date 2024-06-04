
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

df = pd.read_csv('./data/training_set.csv')
df = df.drop(columns=['StationId', 'Datetime'], axis=1)

print('Number of missing values in the dataset: ', df.isnull().sum().sum())

# Impute missing values using IterativeImputer
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=4,max_depth=10, bootstrap=True, max_samples=0.5, n_jobs=2,),max_iter=10, random_state=0, verbose=2)
# imputer = IterativeImputer(estimator=HistGradientBoostingRegressor(early_stopping='auto'),max_iter=5, random_state=0, verbose=2)

df_imputed = imputer.fit_transform(df)

df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

df_imputed.round(3)

df_imputed.to_csv('./data/training_set_imputed_RandomForest.csv', index=False)
# df_imputed.to_csv('./data/training_set_imputed_HistGradientBoostingRegressor.csv', index=False)
