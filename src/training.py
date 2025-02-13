import os
import joblib
import logging
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from utils import calcular_ganancia

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Obtener la ruta base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rutas dinámicas
data_path = os.path.join(BASE_DIR, "data", "df_processed_balanced.csv")
model_dir = os.path.join(BASE_DIR, "models")
eval_results_path = os.path.join(BASE_DIR, "data", "evaluation_results.csv")
conf_matrix_dir = os.path.join(BASE_DIR, "data", "conf_matrices")

# Crear carpeta para matrices de confusión
os.makedirs(conf_matrix_dir, exist_ok=True)

# Cargar los datos de prueba
df_test = pd.read_csv(data_path)
X_test = df_test.drop(columns=['fraude', 'monto'])
y_test = df_test['fraude']
monto_test = df_test['monto']

# Obtener modelos guardados
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

# Evaluar cada modelo
results = []
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = model_file.replace(".pkl", "").replace("_", " ")

    # Cargar modelo
    model = joblib.load(model_path)
    logging.info(f"Evaluando modelo: {model_name}")

    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    ganancia = calcular_ganancia(y_test, y_pred, monto_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Guardar resultados
    results.append({
        "Modelo": model_name,
        "AUC-ROC": round(auc_roc, 4),
        "Ganancia estimada": round(ganancia, 2),
        "Precision 0": round(report["0"]["precision"], 4),
        "Recall 0": round(report["0"]["recall"], 4),
        "Precision 1": round(report["1"]["precision"], 4),
        "Recall 1": round(report["1"]["recall"], 4)
    })

    # Guardar matriz de confusión
    conf_matrix_path = os.path.join(conf_matrix_dir, f"{model_name.replace(' ', '_')}_conf_matrix.csv")
    pd.DataFrame(conf_matrix).to_csv(conf_matrix_path, index=False)
    logging.info(f"Matriz de confusión guardada en {conf_matrix_path}")

    # Log de métricas
    logging.info(f"AUC-ROC: {auc_roc:.4f}")
    logging.info(f"Ganancia estimada: {ganancia:.2f}")
    logging.info("Matriz de confusión:")
    logging.info(conf_matrix)

# Guardar resultados en CSV
os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(eval_results_path, index=False)
logging.info(f"Resultados de evaluación guardados en {eval_results_path}")
