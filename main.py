from src.load_data import load_csv
from src.eda import explore_data
from src.preprocessing import (
    clean_data,
    transform_dates,
    prepare_features,
    aggregate_monthly_sales,
    encode_subcategory_column,
    generate_2025_input
)
from src.modeling import (
    train_and_evaluate,
    train_gb_with_gridsearch,
    train_rf_with_gridsearch,
    train_dt_with_gridsearch
)
import matplotlib.pyplot as plt

# Función para predecir 2025
def predict_sales_2025(model, monthly_data):
    X_2025 = generate_2025_input(monthly_data)
    predictions = model.predict(X_2025)
    X_2025["Predicted Sales"] = predictions
    return X_2025

# Función para graficar comparaciones
def plot_total_sales_by_subcategory(predictions_dict, label_encoder):
    plt.figure(figsize=(12, 6))

    for name, df in predictions_dict.items():
        grouped = df.groupby("Sub-Category")["Predicted Sales"].sum()
        subcat_labels = label_encoder.inverse_transform(grouped.index)
        plt.plot(subcat_labels, grouped.values, marker='o', label=name)

    plt.xticks(rotation=45)
    plt.ylabel("Ventas Totales Predichas (USD)")
    plt.xlabel("Subcategoría")
    plt.title("Comparación de ventas predichas por subcategoría en 2025")
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/sales_comparison_by_subcategory.png')
    plt.show()

def plot_mae_rmse(mae_dict, rmse_dict):
    import numpy as np

    modelos = list(mae_dict.keys())
    x = np.arange(len(modelos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, mae_dict.values(), width, label='MAE', color='skyblue')
    bars2 = ax.bar(x + width/2, rmse_dict.values(), width, label='RMSE', color='salmon')

    ax.set_ylabel('Error')
    ax.set_title('Comparación de MAE y RMSE por modelo')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()

    # Etiquetas sobre barras
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 50, f"{yval:.0f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('plots/mae_rmse_comparison.png')
    plt.show()

# Programa principal
def main():
    # Cargar y preparar datos
    df = load_csv()
    df = clean_data(df)
    df = transform_dates(df)

    # Agregar ventas por subcategoría/mes
    monthly_data = aggregate_monthly_sales(df)
    monthly_data, label_encoder = encode_subcategory_column(monthly_data)

    # Preparar features
    X, y = prepare_features(monthly_data, target_column="Sales")

    # Entrenar modelos - Se muestran graficos de predicción de ventas
    model_lr, mae_lr, rmse_lr = train_and_evaluate(X, y)
    model_rf, mae_rf, rmse_rf = train_rf_with_gridsearch(X, y)
    model_gb, mae_gb, rmse_gb = train_gb_with_gridsearch(X, y)
    model_dt, mae_dt, rmse_dt = train_dt_with_gridsearch(X, y)


    # Predicciones para 2025 - se muestran graficos de predicción de ventas para 2025
    predictions_LR_2025 = predict_sales_2025(model_lr, monthly_data)
    predictions_RF_2025 = predict_sales_2025(model_rf, monthly_data)
    predictions_GB_2025 = predict_sales_2025(model_gb, monthly_data)
    predictions_DT_2025 = predict_sales_2025(model_dt, monthly_data)

    mae_dict = {
    "Linear Reg.": mae_lr,
    "Random Forest": mae_rf,
    "Gradient Boosting": mae_gb,
    "Decision Tree": mae_dt
    }

    rmse_dict = {
        "Linear Reg.": rmse_lr,
        "Random Forest": rmse_rf,
        "Gradient Boosting": rmse_gb,
        "Decision Tree": rmse_dt
    }

    # Imprimir algunas predicciones
    print("\n Predicciones Regresión lineal de ventas por subcategoría y mes en 2025:")
    print(predictions_LR_2025.head(12))
    print("\n Predicciones Random Forest de ventas por subcategoría y mes en 2025:")
    print(predictions_RF_2025.head(12))
    print("\n Predicciones Gradient Boosting de ventas por subcategoría y mes en 2025:")
    print(predictions_GB_2025.head(12))
    print("\n Predicciones Decision Tree de ventas por subcategoría y mes en 2025:")
    print(predictions_DT_2025.head(12))

    # Visualizar comparación de modelos
    predictions_dict = {
        "Linear Reg.": predictions_LR_2025,
        "Random Forest": predictions_RF_2025,
        "Gradient Boosting": predictions_GB_2025,
        "Decision Tree": predictions_DT_2025
    }

    plot_total_sales_by_subcategory(predictions_dict, label_encoder)
    plot_mae_rmse(mae_dict, rmse_dict)

if __name__ == "__main__":
    main()
