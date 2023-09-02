"""
Erika Marlene García Sánchez
A01745158
Modelo k-NN para asignarles categorías a las variables
dependiendo de diferentes características.
El data set es Breast Cancer Wisconsin y es para
predecir si las células cancerosas son benignas o malignas.
2 significa benigno y 4 maligno.
"""
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for more aesthetically data visualization
import numpy as np  # linear algebra
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("./tumor.csv")

# ---------------- PREPROCESAMIENTO ---------------
# checando si hay valores faltantes (no hay)
df.isnull().sum()
# checando si hay valores `na` en el dataframe (no hay)
df.isna().sum()
# Encontrar duplicados
duplicates = df[df.duplicated()]
# Eliminar los duplicados
df = df.drop_duplicates()
# Separando la data en X e y
X = df.drop(["Sample code number", 'Class'], axis=1)
y = df['Class']


def comprobacion_generalizacion():
    """
    La función realiza pruebas de generalización dividiendo los datos en
    conjuntos de entrenamiento y pruebas, escalando las características,
    y luego aplicando el algoritmo de los k-vecinos más cercanos.
    """
    for i in range(2):
        print("\n-------------------- CORRIDA nº", i+1, "--------------------")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None)
        cols = X_train.columns
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=[cols])
        X_test = pd.DataFrame(X_test, columns=[cols])
        print(knn(9, X_train, y_train, X_test, y_test))


def euclidean_distance(test_point, train_point):
    """
    La función calcula la distancia euclidiana entre dos puntos en
    un espacio multidimensional.

    :param test_point: Un array numpy que representa las coordenadas del
                        punto de prueba.
    :param train_point: El parámetro train_point representa un punto en un
                        conjunto de datos de entrenamiento. Puede ser un solo
                        punto de datos o una fila en un conjunto de datos que
                        contiene múltiples puntos.
    :return: la distancia euclidiana entre el test_point y el train_point.
    """
    distance = math.sqrt(np.sum((test_point - train_point) ** 2))
    return distance


# Función que simula el modelo k-NN
def knn(k, X_train, y_train, X_test, y_test):
    """
    La función `knn` implementa el algoritmo de los k-vecinos más cercanos
    para clasificación y evalúa su rendimiento utilizando diversas métricas.

    - `k`: El parámetro "k" representa el número de vecinos más cercanos a
    considerar al realizar predicciones en el algoritmo de los k-vecinos
    más cercanos. Determina cuántos vecinos se utilizarán para
    clasificar un nuevo punto de datos.

    - `X_train`: `X_train` es un DataFrame de pandas que contiene las
    características de los datos de entrenamiento. Cada fila representa un
    punto de datos, y cada columna representa una característica de
    ese punto de datos.

    - `y_train`: El parámetro `y_train` es la variable objetivo para el
    conjunto de entrenamiento. Contiene las etiquetas o categorías verdaderas
    correspondientes a las características en el conjunto
    de entrenamiento `X_train`.

    - `X_test`: `X_test` es un DataFrame de pandas que contiene las
    características de los datos de prueba. Cada fila representa una muestra
    de prueba, y cada columna representa una característica de esa muestra.
    El DataFrame debe tener el mismo número de columnas que el
    DataFrame `X_train`.

    - `y_test`: El parámetro `y_test` son las etiquetas o categorías
    verdaderas para los datos de prueba. Es una lista o arreglo que contiene
    las categorías reales de las muestras de prueba.

    La función devuelve una lista de categorías asignadas
    para los datos de prueba.
    """
    assigned_categories = []
    for _, row in X_test.iterrows():
        complete_row = np.array([])
        for _, test_point in row.items():
            if test_point is not None:
                complete_row = np.append(complete_row, test_point)

        train_distances = []
        for index_train, row_train in X_train.iterrows():
            train_complete_row = np.array([])
            for _, train_point in row_train.items():
                if train_point is not None:
                    train_complete_row = np.append(train_complete_row,
                                                   train_point)
            distance = euclidean_distance(complete_row, train_complete_row)
            train_distances.append((distance, y_train.iloc[index_train]))

        train_distances.sort(key=lambda x: x[0])
        nearests = train_distances[:k]

        twos = sum(1 for _, label in nearests if label == 2)
        fours = sum(1 for _, label in nearests if label == 4)

        if twos > fours:
            assigned_categories.append(2)
        elif twos < fours:
            assigned_categories.append(4)

    correct_predictions = 0
    incorrect_predictions = 0
    misclassified_indices = []

    for i, (predicted, actual) in enumerate(zip(assigned_categories,
                                                y_test.to_list())):
        if predicted == actual:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
            misclassified_indices.append(i)

    print("\nPrecisión del modelo k-NN score",
          accuracy_score(y_test, assigned_categories))
    print()
    print(classification_report(y_test, assigned_categories))
    print("El porcentaje de predicciones incorrectas es de:",
          ((incorrect_predictions * 100)/len(assigned_categories)), "%")
    print("Índices de predicciones incorrectas:", misclassified_indices, "\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, assigned_categories)
    df1 = pd.DataFrame(columns=["2", "4"],
                       index=["benigno", "maligno"], data=cm)

    f, ax = plt.subplots(figsize=(2, 2))

    sns.heatmap(df1, annot=True, cmap="Greens", fmt='.0f',
                ax=ax, linewidths=5, cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predicted Label")
    plt.xticks(size=10)
    plt.yticks(size=10, rotation=0)
    plt.ylabel("True Label")
    plt.title("Confusion Matrix", size=10)
    plt.show()
    return assigned_categories


comprobacion_generalizacion()
