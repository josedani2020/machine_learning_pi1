import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import json
import pyodbc
from datetime import datetime
import time

def load_config():
    with open("config.json", "r") as file:
        config = json.load(file)
    return config

def save_metrics(algorithm_name, metrics):
    with open(f"{algorithm_name}_metrics_results.txt", "a") as file:
        file.write(f"{datetime.now()}: {algorithm_name} Metrics\n")
        for metric, value in metrics.items():
            file.write(f"{metric}: {value}\n")
        file.write("\n")

def main():
    start_time = time.time()
    config = load_config()
    connection_string = f"DRIVER={{SQL Server}};SERVER={config['server']};DATABASE={config['database']};UID={config['username']};PWD={config['password']}"
    connection = pyodbc.connect(connection_string)

    df_ventas = pd.read_sql("SELECT * FROM dbo.VENTAS", connection)
    df_equivalencias = pd.read_sql("SELECT * FROM dbo.EQUIVALENCIAS", connection)

    df_ventas['KEY'] = df_ventas['COD_DISTRIBUIDORA'].astype(str) + df_ventas['COD_PRODUCTO'].astype(str)
    df_equivalencias['KEY'] = df_equivalencias['COD_DISTRIBUIDORA'].astype(str) + df_equivalencias['COD_PROD_DISTRIBUIDOR'].astype(str)
    df_ventas['Homologado'] = np.where(df_ventas['KEY'].isin(df_equivalencias['KEY']), 'Homologados', 'No Homologados')

    X = pd.get_dummies(df_ventas[['KEY']])
    y = df_ventas['Homologado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)  # Enable probability for ROC AUC
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if y.nunique() == 2 else None  # Calculate probabilities only for binary classification

    # Evaluate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1-Score': f1_score(y_test, y_pred, average='macro'),
        'ROC AUC': roc_auc_score(label_binarize(y_test, classes=['Homologados', 'No Homologados']), label_binarize(y_pred, classes=['Homologados', 'No Homologados'])) if y.nunique() == 2 else 'N/A'
    }

    # Save metrics to file
    save_metrics("SVM", metrics)

    # Insert predicted results into database
    with connection.cursor() as cursor:
        for index, row in df_ventas.loc[X_test.index].iterrows():
            cursor.execute("INSERT INTO RESULTADOS_CLASIFICACION_SVM (COD_DISTRIBUIDORA, COD_PRODUCTO, Predicted_Homologado, Fecha_Registro) VALUES (?, ?, ?, ?)", 
                           (row['COD_DISTRIBUIDORA'], row['COD_PRODUCTO'], row['Predicted_Homologado'], datetime.now()))
        connection.commit()

    connection.close()
    total_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

if __name__ == "__main__":
    main()
