import seaborn as sns
import matplotlib.pyplot as plt

def explore_data(df):
    print("Información general:")
    print(df.info())
    print("\nDescripción estadística:")
    print(df.describe())

    print("\nValores nulos:")
    print(df.isnull().sum())

    sns.histplot(df["Sales"], kde=True)
    plt.title("Distribución de Ventas")
    plt.xlabel("Ventas (USD)")
    plt.ylabel("Frecuencia")
    plt.show()

    if df.select_dtypes(include='number').shape[1] > 1:
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        plt.title("Correlación entre variables numéricas")
        plt.show()
