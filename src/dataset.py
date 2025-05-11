import kagglehub
import os
import shutil

# Asegúrate de que la carpeta 'data' exista
os.makedirs('data', exist_ok=True)

# Descargar el dataset sin especificar el path y moverlo después
path = kagglehub.dataset_download("rohitsahoo/sales-forecasting")

# Mover el dataset a la carpeta 'data'
shutil.move(path, "data/sales-forecasting")

print("Path to dataset files:", os.path.join('data', 'sales-forecasting'))
