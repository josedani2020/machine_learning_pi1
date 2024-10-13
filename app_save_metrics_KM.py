import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
import pyodbc
from datetime import datetime
import time

def load_config():
    with open("config.json", "r") as file:
        config = json.load(file)
    return config

def save_metrics(metrics):
    with open("metrics_results.txt", "a") as file:
        for metric_name, value in metrics.items():
            file.write(f"{datetime.now()}: K-Means - {metric_name}: {value}\n")

def main():
    start_time = time.time()
    config = load_config()
    connection_string = f"DRIVER={{SQL Server}};SERVER={config['server']};DATABASE={config['database']};UID={config['username']};PWD={config['password']}"
    connection = pyodbc.connect(connection_string)

    df_ventas = pd.read_sql("SELECT * FROM dbo.VENTAS", connection)
    df_ventas_encoded = pd.get_dummies(df_ventas[['COD_DISTRIBUIDORA', 'COD_PRODUCTO']])
    scaler = StandardScaler()
    df_ventas_scaled = scaler.fit_transform(df_ventas_encoded)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df_ventas_scaled)
    clusters = kmeans.predict(df_ventas_scaled)

    metrics = {
        'Silhouette Score': silhouette_score(df_ventas_scaled, clusters),
        'Calinski-Harabasz Score': calinski_harabasz_score(df_ventas_scaled, clusters),
        'Davies-Bouldin Score': davies_bouldin_score(df_ventas_scaled, clusters)
    }
    save_metrics(metrics)

    connection.close()
    total_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

if __name__ == "__main__":
    main()
