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


#%%
"""
Let's start with a vanilla regressor and a simple pipeline"
"""

df = pd.read_parquet('rents_clean.parquet')
columns = ['prezzo', 'bagni', 'stanze', 'superficie']
df = df[columns].dropna()
df = df.loc[df['prezzo'] < 10000]

regressor = RandomForestRegressor()

REGRESSOR_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
}

regressor.set_params(**REGRESSOR_PARAMETERS)
"""
The notation ** is used to unpack a dictionary. 
In this case, the dictionary is PARAMETERS, which is a dictionary of hyperparameters for the regressor. 
The set_params method of the regressor object takes a dictionary of hyperparameters and sets them. 
The ** operator unpacks the dictionary and passes the key-value pairs as arguments to the set_params method. This is equivalent to:
my_regressor.set_params(n_estimators=100, max_depth=None, min_samples_split=2)
This will be usefull later to pass the hyperparameters to the pipeline automatically.
"""

X = df.drop(columns=['prezzo'])
y = df['prezzo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

my_regressor.fit(X_train, y_train)



#%% NESTING 1 - PIPELINE

"""
Let's now implement the pipeline with the following steps:
1. Impute missing values in numerical features with the mean
2. Scale numerical features
3. Impute missing values in categorical features with the most frequent value
4. One-hot encode categorical features
5. Combine the 2 pipelines with ColumnTransformer
6. Fit the pipeline
Note that I'm only using 3 features for simplicity. So we...
"""

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

"""
Since the regressor is a step of the pipeline, it is a nested estimator. 
This means that we need to use the __ notation to set its parameters. 
The double underscore ('__') is a standard Scikit-learn convention used for setting parameters of nested objects. 
Here, 'regressor' is the name of the step in our pipeline. 
By using 'regressor__<parameter_name>', we are specifying a parameter for the 'regressor' step. 
This is essential when working with nested objects like pipelines.
It goes like this:
"""

REGRESSOR_PARAMETERS = {
    "regressor__n_estimators": 100,
    "regressor__max_depth": None,
    "regressor__min_samples_split": 2,
}
pipeline.set_params(**REGRESSOR_PARAMETERS)

"""
where 'regressor' is the name of the estimator. When scikit-learn sees the __ notation,
it knows that the hyperparameter is nested inside the regressor. It is equivalent to writing:
my_regressor = RandomForestRegressor()
my_regressor.set_params(n_estimators=100, max_depth=None, min_samples_split=2)
where my_regressor is the regressor step of the pipeline, and 'regressor' is the name of the step.
This way, we can pass the PARAMETERS dictionary to the pipeline:
"""

"""
and then fit the pipeline as usual:"""
# fit pipeline
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

"""
We can check the hyperparameters of the pipeline:
"""

params = pipeline.get_params()
for key, value in params.items():
    print(f"{key}: {value}")


#%% gridsearch
"""
The same way can be used to perform gridsearch. We just need to pass the PARAMETERS_GRID dictionary to GridSearchCV:
"""

GRID_PARAMETERS = {
    "regressor__n_estimators": [50, 100],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(pipeline, 
                           GRID_PARAMETERS, 
                           cv=2, verbose=1, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)


#%% NESTING 2 - KERAS WRAPPER
"""
To incorporate a TensorFlow model into our Scikit-learn pipeline, we need to use a "wrapper". 
A wrapper acts as a bridge between the Scikit-learn API and TensorFlow's. 
The Scikeras KerasRegressor wrapper inherits from BaseEstimator and implements 
the methods fit, predict, score, set_params and others, 
and then passes the parameters to the neural networl.
Basically, it allows us to create a NN by using the Scikit-learn syntax. 
Let's first create a neural network with TensorFlow:
"""

"""
To use the wrappers, you must specify how to build the Keras model. The wrappers support both Sequential and Functional API models. There are 2 basic options to specify how the model is built:

Prebuilt model: pass an existing Keras model object to the wrapper, which will be copied to avoid modifying the existing model. You must pass the prebuilt model via the 'model' parameter when initializing the wrapper.
Dynamically built model: pass a function that returns a model object. The model will not be built until fit is called. More details below.
"""

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def model_factory(
    num_features, 
    num_labels, 
    hidden_units, 
    dropout_rate=None, 
    optimizer='adam', 
    loss='mse', 
    metrics=None
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

"""
Now we define the regressor. 
We need to distinguish the parameters of the wrapper from the parameters of the build_fn (our 'model_factory).
KerasRegressorWrapper has the following parameters: 
- build_fn: the function that builds the neural network
- loss: the loss function
- Other metrics that we don't use here

build_fn is a function that creates the NN architecture, and it takes the following parameters:
- num_features: the number of features of the dataset
- num_labels: the number of labels of the dataset
- hidden_units: a tuple of integers, specifying the number of units of each dense layer
- dropout_rate: the dropout rate of the dropout layers
- other parameters that we don't use here

The tricky part is that build_fn will be considered as a nested parameter of the regressor,
while the regressor is already a nested parameter of the pipeline.
So we need to use the __ notation twice.
NB: KerasRegressor uses the name 'model' as the name of the build_fn parameter.
Therefore, our PARAMETERS need to have the following structure:
<name of the regressor>__<name of the build_fn>__<name of the parameter>
"""
from scikeras.wrappers import KerasRegressor

regressor = KerasRegressor(build_fn=model_factory(**REGRESSOR_PARAMETERS), 
                           loss="mse",
                           )

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

input_shape = preprocessor.fit_transform(X_train).shape[1]

PARAMETERS = {
    # regressor parameters
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

pipeline.set_params(**REGRESSOR_PARAMETERS)
pipeline.get_params()['regressor'].__dict__
"""
After setting the parameters, we can fit the pipeline as usual. 
The REGRESSOR_PARAMETERS  will be passed only during the fit
"""
pipeline.fit(X_train, y_train)

#%% GRIDSEARCH 

"""
We will pass iWe are finally able to compare a classical regressor from scikit-learn with a neural network from tensorflow. 
Let's re-write the pipeline, but now we don't pass a model to the regressor step and we leave it as a placeholder (None).
t as a parameter to GridSearchCV, and GridSearchCV will take care of fitting the pipeline with the correct model.
"""

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor_placeholder', None)
])

GRID_PARAMETERS = [
    {
        "regressor_placeholder": [KerasRegressor()],
        "regressor_placeholder__model__hidden_units": (128,)*10,
        "regressor_placeholder__model__dropout_rate": [0.1, 0.2],
        "regressor_placeholder__epochs": [1],
        "regressor_placeholder__batch_size": [128],
        "regressor_placeholder__verbose": [1],
        "regressor_placeholder__metrics": ["mae"],
    },
    {
        "regressor_placeholder": [RandomForestRegressor()],
        "regressor_placeholder__n_estimators": [50, 100], 
        "regressor_placeholder__max_depth": [None],
        "regressor_placeholder__min_samples_split": [2],
    }
]

"""
NB: Note that when we pass the Random forest, we only use **regressor__** as a prefix. 
However, when we pass  the KerasRegressorWrapper, we use **regressor_placeholder__model__** as a prefix.
Basically, we are telling the pipeline:
- go to the model_placeholder step
    - if the model is random forest (or any other scikit-learn regressor), pass the hyperparameters to the regressor directly
    - if the model the Keras Wrapper, pass the hidden_units parameter to the build_model function with the double __ notation
"""

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

"""
SUMMARY:

In this notebook, we delved deep into the world of Scikit-learn pipelines and nested parameter settings:

- First Level (regressor__...): Introduced by the Scikit-learn Pipeline. 
Parameters for pipeline steps are set using the step name followed by two underscores.
- Second Level (model__...): Specific to the KerasRegressor from scikeras. 
When setting parameters for the model-building function (build_fn), we use 'model__'.

This nested parameter setting allows us to seamlessly combine preprocessing steps, 
classical Scikit-learn regressors, and complex TensorFlow models into a unified workflow.
"""
