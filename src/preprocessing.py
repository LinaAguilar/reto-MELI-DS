import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.combine import SMOTETomek
import pandas as pd
import scipy.stats as stats

#cargar los datos desde csv
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "fraudDataset.csv")
df = pd.read_csv(data_path)

# Imputar con la mediana
df_processed = df.copy()
df_processed['b'].fillna(df_processed['b'].median(), inplace=True) #por tener distribución sesgada
df_processed['c'].fillna(df_processed['c'].median(), inplace=True) #altamente sesgada con outliers
df_processed['d'].fillna(df_processed['d'].median(), inplace=True) #sesgada
df_processed['f'].fillna(df_processed['f'].median(), inplace=True) #sesgada
df_processed['l'].fillna(df_processed['l'].median(), inplace=True) #sesgada
df_processed['m'].fillna(df_processed['m'].median(), inplace=True) #sesgada

df_processed['o'].fillna('UN', inplace=True) #tiene muchos valores nulos luego llenamos con UNKNOWN
df_processed['g'].fillna(df_processed['g'].mode()[0], inplace=True) #categórica luego llenamos con la moda

# convertir fecha a date_time y extraer características
df_processed['fecha'] = pd.to_datetime(df_processed['fecha'])
df_processed['dia'] = df_processed['fecha'].dt.day
df_processed['mes'] = df_processed['fecha'].dt.month
df_processed['hora'] = df_processed['fecha'].dt.hour



# Normalización estándar
scaler = StandardScaler()
df_processed[['normalized_monto', 'score', 'b', 'k']] = scaler.fit_transform(df_processed[['monto', 'score', 'b', 'k']])

# Transformación logarítmica para distribuciones sesgadas con valores positivos
df_processed['c'] = np.log1p(df_processed['c'])
df_processed['m'] = np.log1p(df_processed['m'])

# Escalado robusto por la presencia de outliers y valores negativos
robust_scaler = RobustScaler()
df_processed[['l','f']] = robust_scaler.fit_transform(df_processed[['l','f']])

# One-Hot Encoding para variables categóricas
df_processed = pd.get_dummies(df_processed, columns=['a', 'g', 'o', 'p'], drop_first=True)

# Crear una tabla de contingencia entre 'j' y 'fraude'
contingency_table = pd.crosstab(df_processed['j'], df_processed['fraude'])

# Calcular el estadístico de Chi-Cuadrado y el coeficiente de Cramér's V
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
n = contingency_table.sum().sum()
phi2 = chi2 / n
r, k = contingency_table.shape
cramers_v = np.sqrt(phi2 / min((k-1), (r-1)))

print(f"Coeficiente de Cramér's V: {cramers_v:.4f}")

df_processed.drop(columns=['j','fecha'], inplace=True)


#df_processed.to_csv("../data/df_processed.csv", index=False)

# Separar las características (X) y la variable objetivo (y)
X = df_processed.drop(columns=['fraude'])
y = df_processed['fraude']

# Aplicar SMOTE + Tomek Links
smt = SMOTETomek(random_state=42)
X_balanced, y_balanced = smt.fit_resample(X, y)

# Crear un nuevo DataFrame balanceado
df_processed_balanced = pd.DataFrame(X_balanced, columns=X.columns)
df_processed_balanced['fraude'] = y_balanced

# Verificar el balance de clases
print(df_processed_balanced['fraude'].value_counts())

data_dir = os.path.join(BASE_DIR, "data")
processed_file = os.path.join(data_dir, "df_processed_balanced.csv")
df_processed.to_csv(processed_file, index=False)