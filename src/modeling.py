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

    plt.figure()
    plt.scatter(y_test, predictions, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Linear Regression: Predicción vs Real")
    plt.tight_layout()
    plt.savefig('plots/linear_regression_eval.png')
    plt.show()

    return model, mae, rmse

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
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f" Mejor combinación de parámetros: {grid_search.best_params_}")
    print(f" MAE promedio (CV): {-grid_search.best_score_:.2f}")

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    plt.figure()
    plt.scatter(y, best_model.predict(X), alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Random Forest con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.savefig('plots/random_forest_eval.png')
    plt.show()

    return best_model, -grid_search.best_score_, np.sqrt(mean_squared_error(y, best_model.predict(X)))

def train_gb_with_gridsearch(X, y):
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
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f" Mejor combinación de parámetros: {grid_search.best_params_}")
    print(f" MAE promedio (CV): {-grid_search.best_score_:.2f}")

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    plt.figure()
    plt.scatter(y, predictions, alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Gradient Boosting con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.savefig('plots/gradient_boosting_eval.png')
    plt.show()

    return best_model, -grid_search.best_score_, np.sqrt(mean_squared_error(y, best_model.predict(X)))

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

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    plt.figure()
    plt.scatter(y, best_model.predict(X), alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Ventas reales (USD)")
    plt.ylabel("Ventas predichas (USD)")
    plt.title("Decision Tree con GridSearch: Predicción vs Real")
    plt.tight_layout()
    plt.savefig('plots/decision_tree_eval.png')
    plt.show()

    return best_model, -grid_search.best_score_, np.sqrt(mean_squared_error(y, best_model.predict(X)))


