#%%
import pandas as pd
import numpy as np
#import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import warnings
warnings.filterwarnings('ignore')




df = pd.read_parquet('rents_clean.parquet')
columns = ['prezzo', 'bagni', 'stanze', 'superficie']
df = df[columns].dropna()
df = df.loc[df['prezzo'] < 10000]

my_regressor = RandomForestRegressor()

REGRESSOR_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
}

my_regressor.set_params(**REGRESSOR_PARAMETERS)


numerical_features = ['superficie']
categorical_features = ['bagni', 'stanze']
target = 'prezzo'

num_imputer = SimpleImputer(strategy='mean')  
cat_imputer = SimpleImputer(strategy='most_frequent')
hot_encoder = OneHotEncoder(handle_unknown='ignore')

# create numerical vs categorical pipelines withouth pandas selector
numerical_pipeline = Pipeline([
    ('imputer', num_imputer),
    ('scaler', StandardScaler()) 
])

categorical_pipeline = Pipeline([
    ('imputer', cat_imputer),
    ('encoder', hot_encoder)
])

# final preprocessor
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
], remainder='drop')

# combine categorical and numerical pipelines
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
], remainder='drop')

# final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', my_regressor)
])

REGRESSOR_PARAMETERS = {
    "regressor__n_estimators": 100,
    "regressor__max_depth": None,
    "regressor__min_samples_split": 2,
}
pipeline.set_params(**REGRESSOR_PARAMETERS)


# fit pipeline
# pipeline.fit(X_train, y_train)
# pipeline.score(X_test, y_test)


params = pipeline.get_params()
for key, value in params.items():
    print(f"{key}: {value}")




#%%
X = df.drop(columns=['prezzo'])
y = df['prezzo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def model_factory(
    num_features, num_labels, hidden_units, dropout_rate=None, optimizer='adam', loss='mse', metrics=None
):

    features = Input(shape=(num_features,), name="features")

    layers = features
    for idx, units in enumerate(hidden_units):
        layers = Dense(units, activation="relu", name=f"dense_{idx}")(layers)
        if dropout_rate:
            layers = Dropout(dropout_rate, name=f"dropout_{idx}")(layers)

    labels = Dense(num_labels, name="labels")(layers)

    model = Model(inputs=features, outputs=labels)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


from scikeras.wrappers import KerasRegressor

regressor = KerasRegressor(loss="mse")


#%%
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

input_shape = preprocessor.fit_transform(X_train).shape[1]

PARAMETERS = {
    # NN parameters
    "regressor__model": model_factory,
    "regressor__model__num_features": input_shape,
    "regressor__model__num_labels": 1,
    "regressor__model__hidden_units": (128,)*10,
    "regressor__model__dropout_rate": 0.1,

    # training parameter
    "regressor__epochs": 1,
    "regressor__batch_size": 128,
    "regressor__verbose": 1,
    "regressor__metrics": ["mae"],
    "regressor__optimizer": "adam",
    "regressor__loss": "mse",
}

pipeline.set_params(**PARAMETERS)
pipeline.get_params()['regressor'].__dict__


#%%
"""
After setting the parameters, we can fit the pipeline as usual:
The Model parameters will be passed only during the fit
"""
pipeline.fit(X_train, y_train)

#%%
constructed_model = pipeline.named_steps['regressor'].model_
constructed_model.summary()

# inspect model loss and metrics
history = pipeline.named_steps['regressor'].history_
history.history



#%% GRIDSEARCH 

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor_placeholder', None)
])

GRID_PARAMETERS = [
    {
        "regressor_placeholder": [KerasRegressor()],
        "regressor_placeholder__model__hidden_units": (128,)*10,
        "regressor_placeholder__model__dropout_rate": [0.1, 0.2],
    },
    {
        "regressor_placeholder": [RandomForestRegressor()],
        "regressor_placeholder__n_estimators": [50, 100], 
        "regressor_placeholder__max_depth": [None],
        "regressor_placeholder__min_samples_split": [2],
    }
]

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=GRID_PARAMETERS, 
                           cv=2, 
                           verbose=1, 
                           n_jobs=-1, 
                           scoring='neg_mean_absolute_error')

grid_search.fit(X_train, y_train)

print("Best hyperparameters:\n", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

"""
Finally, we can directly use the best model to predict the test set with grid_search.best_estimator_
"""

best_model = grid_search.best_estimator_
grid_search.best_estimator_.fit(X_train, y_train)
grid_search.best_estimator_.score(X_test, y_test)


#%%
