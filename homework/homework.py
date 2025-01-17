#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score
import os
import pickle
import gzip
import json

# Paso 1.
# Preprocese los datos.

def preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, test_df]:
        df['Age'] = 2021 - df['Year']
        df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    
    return train_df, test_df

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

def split_data(train_df, test_df):
    x_train = train_df.drop('Present_Price', axis=1)
    y_train = train_df['Present_Price']
    x_test = test_df.drop('Present_Price', axis=1)
    y_test = test_df['Present_Price']
     
    return x_train, y_train, x_test, y_test

# Paso 3.
# Cree un pipeline para el modelo de clasificación.
def build_pipeline():
    numeric_features = ['Selling_Price', 'Driven_kms', 'Age', 'Owner']
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'feature_selection__k': [5, 10, 'all']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_absolute_error')
    grid_search.fit(x_train, y_train)
    
    return grid_search

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
def save_model(model):    
    models_dir = 'files/models'
    os.makedirs(models_dir, exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)

# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
def calculate_metrics(model, x_train, y_train, x_test, y_test):
    metrics = []
    
    for x, y, dataset_name in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
        y_pred = model.predict(x)
        metrics.append({
            'type': 'metrics',
            'dataset': dataset_name,
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mad': median_absolute_error(y, y_pred)
        })
    
    return metrics

def save_metrics(metrics):
    metrics_path = "files/output/metrics.json"
    os.makedirs("files/output", exist_ok=True)
    with open(metrics_path, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric, ensure_ascii=False))
            f.write('\n')

def main():
    train_path = 'files/input/train_data.csv.zip'
    test_path = 'files/input/test_data.csv.zip'
    
    train_df, test_df = preprocess_data(train_path, test_path)
    x_train, y_train, x_test, y_test = split_data(train_df, test_df)
    
    pipeline = build_pipeline()
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    save_model(model)
    
    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)   

if __name__ == "__main__":
    main()