import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import pyodbc
from datetime import datetime
import time

def load_config():
    with open("config.json", "r") as file:
        config = json.load(file)
    return config

def main():
    start_time = time.time()
    config = load_config()
    connection_string = f"DRIVER={{SQL Server}};SERVER={config['server']};DATABASE={config['database']};UID={config['username']};PWD={config['password']}"
    connection = pyodbc.connect(connection_string)

    # Carga de datos
    load_data_start = time.time()
    df_ventas = pd.read_sql("SELECT * FROM dbo.VENTAS", connection)
    load_data_end = time.time()
    print(f"Tiempo de carga de datos: {load_data_end - load_data_start:.2f} segundos")

    # Procesamiento de datos
    process_data_start = time.time()
    df_ventas_encoded = pd.get_dummies(df_ventas[['COD_DISTRIBUIDORA', 'COD_PRODUCTO']])
    scaler = StandardScaler()
    df_ventas_scaled = scaler.fit_transform(df_ventas_encoded)
    process_data_end = time.time()
    print(f"Tiempo de procesamiento de datos: {process_data_end - process_data_start:.2f} segundos")

    # Modelo de clustering usando K-Means
    ml_model_start = time.time()
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df_ventas_scaled)
    clusters = kmeans.predict(df_ventas_scaled)
    df_ventas['Cluster'] = clusters
    ml_model_end = time.time()
    print(f"Tiempo de modelo de machine learning (K-Means): {ml_model_end - ml_model_start:.2f} segundos")

    # Inserción de resultados de clustering en la base de datos
    insertion_start = time.time()
    cursor = connection.cursor()
    for index, row in df_ventas.iterrows():
        cursor.execute("INSERT INTO RESULTADOS_CLASIFICACION_KM (COD_DISTRIBUIDORA, COD_PRODUCTO, Cluster, Fecha_Registro) VALUES (?, ?, ?, GETDATE())",
                       (row['COD_DISTRIBUIDORA'], row['COD_PRODUCTO'], row['Cluster']))
    connection.commit()
    insertion_end = time.time()
    print(f"Tiempo de inserción en la base de datos: {insertion_end - insertion_start:.2f} segundos")

    connection.close()
    total_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

if __name__ == "__main__":
    main()
