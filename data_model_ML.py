#%%

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

# %%

df = pd.read_csv('real_estate_clean_ML.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.dropna(subset=['property_type', 'price'], axis=0)

#%%

df.columns

#%%

df = df[df['transaction_type'] == 2.]
df = df.drop(['titre', 'date', 'tag_list', 'desc', 'geoloc', 'ref', 'city', 'psqrt_tag', 'terrain_sqrt_tag', 'city_a', 'quartier_a'], axis=1)

#%%

# Build feature/target arrays
X, y = df.drop('price', axis=1), df['price']

# Generate train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1121218)

# Init, fit, score
model = RandomForestRegressor()
_ = model.fit(X_train, y_train)

#%%
print(f"Training score: {model.score(X_train, y_train)}")
print(f"Test score: {model.score(X_test, y_test)}")


# %%

params = {
    'n_estimators' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_depth' : [5, 10, 15, 20, 25, 30, 40, 50, 60]
}
grid = GridSearchCV(model, param_grid=params, cv=5)
grid.fit(X_train, y_train)
grid.best_params_

# %%



# %%



#%%


#%%


#%%


#%%

