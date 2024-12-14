import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження датасету
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Базовий аналіз даних
print(X.describe())
print(f"Пропущені значення: \n{X.isnull().sum()}")

# Візуальний аналіз
def visualize_data(X, y):
    for column in X.columns:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(X[column], kde=True)
        plt.title(f"Гістограма {column}")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=X[column])
        plt.title(f"Boxplot {column}")
        plt.show()

    # Кореляційна матриця
    corr_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Кореляційна матриця")
    plt.show()

visualize_data(X, y)

# Розділення на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Проста лінійна регресія (найбільш корельована ознака)
corr_with_target = pd.Series(np.corrcoef(X, y, rowvar=False)[-1, :-1], index=X.columns)
most_correlated_feature = corr_with_target.idxmax()

lr_simple = LinearRegression()
lr_simple.fit(X_train_scaled[:, [X.columns.get_loc(most_correlated_feature)]], y_train)

# Прогнозування та оцінка
simple_pred = lr_simple.predict(X_test_scaled[:, [X.columns.get_loc(most_correlated_feature)]])
simple_mse = mean_squared_error(y_test, simple_pred)
simple_r2 = r2_score(y_test, simple_pred)
print(f"Проста регресія: MSE={simple_mse}, R²={simple_r2}")

# Множинна лінійна регресія
lr_multiple = LinearRegression()
lr_multiple.fit(X_train_scaled, y_train)

multiple_pred = lr_multiple.predict(X_test_scaled)
multiple_mse = mean_squared_error(y_test, multiple_pred)
multiple_r2 = r2_score(y_test, multiple_pred)
print(f"Множинна регресія: MSE={multiple_mse}, R²={multiple_r2}")

# Регуляризація (Ridge та Lasso)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

ridge_pred = ridge.predict(X_test_scaled)
lasso_pred = lasso.predict(X_test_scaled)

ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"Ridge регресія: MSE={ridge_mse}, R²={ridge_r2}")
print(f"Lasso регресія: MSE={lasso_mse}, R²={lasso_r2}")

# Графіки оцінки моделей
def plot_results(y_test, predictions, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title(title)
    plt.xlabel("Реальні значення")
    plt.ylabel("Передбачені значення")
    plt.show()

plot_results(y_test, simple_pred, "Проста регресія")
plot_results(y_test, multiple_pred, "Множинна регресія")
plot_results(y_test, ridge_pred, "Ridge регресія")
plot_results(y_test, lasso_pred, "Lasso регресія")
input("\nНатисніть Enter, щоб завершити програму...")
