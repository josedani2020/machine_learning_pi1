import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
    df_equivalencias = pd.read_sql("SELECT * FROM dbo.EQUIVALENCIAS", connection)
    load_data_end = time.time()
    print(f"Tiempo de carga de datos: {load_data_end - load_data_start:.2f} segundos")

    # Procesamiento de datos
    process_data_start = time.time()
    df_ventas['KEY'] = df_ventas['COD_DISTRIBUIDORA'].astype(str) + df_ventas['COD_PRODUCTO'].astype(str)
    df_equivalencias['KEY'] = df_equivalencias['COD_DISTRIBUIDORA'].astype(str) + df_equivalencias['COD_PROD_DISTRIBUIDOR'].astype(str)
    df_ventas['Homologado'] = np.where(df_ventas['KEY'].isin(df_equivalencias['KEY']), 'Homologados', 'No Homologados')
    process_data_end = time.time()
    print(f"Tiempo de procesamiento de datos: {process_data_end - process_data_start:.2f} segundos")

    # Modelo de machine learning usando SVM
    ml_model_start = time.time()
    X = pd.get_dummies(df_ventas[['KEY']])
    y = df_ventas['Homologado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar y entrenar el modelo SVC
    model = SVC(kernel='linear')  # Se puede cambiar el kernel si es necesario
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    df_ventas.loc[X_test.index, 'Predicted_Homologado'] = y_pred
    ml_model_end = time.time()
    print(f"Tiempo de modelo de machine learning (SVM): {ml_model_end - ml_model_start:.2f} segundos")

    # Inserción en base de datos
    insertion_start = time.time()
    now = datetime.now()
    with connection.cursor() as cursor:
        for index, row in df_ventas.loc[X_test.index].iterrows():
            cursor.execute("INSERT INTO RESULTADOS_CLASIFICACION_SVM (COD_DISTRIBUIDORA, COD_PRODUCTO, Predicted_Homologado, Fecha_Registro) VALUES (?, ?, ?, ?)", row['COD_DISTRIBUIDORA'], row['COD_PRODUCTO'], row['Predicted_Homologado'], now)
        connection.commit()
    insertion_end = time.time()
    print(f"Tiempo de inserción en la base de datos: {insertion_end - insertion_start:.2f} segundos")

    connection.close()
    total_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

print("ok")

if __name__ == "__main__":
    main()
