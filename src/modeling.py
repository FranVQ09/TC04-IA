from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Linear Regression --> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    plt.scatter(y_test, predictions, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Linear Regression: Predicción vs Real")
    plt.tight_layout()
    plt.show()

def train_rf_with_gridsearch(X, y):
    print("\n Grid Search + Cross-Validation: Random Forest Regressor")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "max_features": ["sqrt", "log2"]
    }

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold CV
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f" Mejor combinación de parámetros: {grid_search.best_params_}")
    print(f" MAE promedio (CV): {-grid_search.best_score_:.2f}")

    # Visualizar predicciones
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    plt.scatter(y, predictions, alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Random Forest con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.show()

def train_with_gridsearch(X, y):
    print("\n Grid Search + Cross-Validation: Gradient Boosting Regressor")

    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }

    model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="neg_mean_absolute_error",
        n_jobs=-1,  # usa todos los núcleos disponibles
        verbose=1
    )

    grid_search.fit(X, y)

    print(f" Mejor combinación de parámetros: {grid_search.best_params_}")
    print(f" MAE promedio (CV): {-grid_search.best_score_:.2f}")

    # También evaluamos con predicciones finales
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    # (opcional) podrías usar test split si querés separar train/test otra vez
    plt.scatter(y, predictions, alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Gradient Boosting con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.show()

def train_dt_with_gridsearch(X, y):
    print("\n Grid Search + Cross-Validation: Decision Tree Regressor")

    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    model = DecisionTreeRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f" Mejor combinación de parámetros: {grid_search.best_params_}")
    print(f" MAE promedio (CV): {-grid_search.best_score_:.2f}")

    # Evaluación gráfica con el mejor modelo
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    plt.scatter(y, predictions, alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Decision Tree con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.show()

