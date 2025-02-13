import pandas as pd

def calcular_ganancia(y_true, y_pred, monto):
    """
    Calcula la ganancia basada en las predicciones:
    - Falso negativo (fraude no detectado) -> pérdida del 100% del monto
    - Transacción aprobada correctamente -> ganancia del 25% del monto
    - Transacción denegada -> sin ganancia
    """
    df_resultado = pd.DataFrame({'fraude_real': y_true, 'fraude_predicho': y_pred, 'monto': monto})

    # Casos donde el fraude real es 1 pero fue predicho como 0 (Falso Negativo)
    perdida = df_resultado[(df_resultado['fraude_real'] == 1) & (df_resultado['fraude_predicho'] == 0)]['monto'].sum()

    # Casos donde se aprueba la transacción (fraude_predicho == 0), se obtiene 25% de ganancia
    ganancia = df_resultado[df_resultado['fraude_predicho'] == 0]['monto'].sum() * 0.25

    # Ganancia total
    ganancia_total = ganancia - perdida

    return ganancia_total
