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

df = pd.read_csv("./tumor.csv")
# print(df.head())

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
# separando X e y en sets de training y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=None)
# print(X_train.head())

"""
Haciendo escalamiento de características para mejorar el rendimiento
del modelo ya que es sensible a la escala
"""

cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
# print(X_train.head())


"""
Función que calcula la distancia euclideana entre los puntos de test y train
Usa todas las columnas: ('Clump Thickness','Uniformity of Cell Size'
,'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size'
, 'Bare Nuclei','Bland Chromatin', 'Normal Nucleoli', 'Mitoses',)
"""


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
def knn(k, X_train, y_train, X_test):
    """
    La función `knn` implementa el algoritmo de los k-vecinos más cercanos para
    predecir las categorías de los datos de prueba basándose en los vecinos más
    cercanos en los datos de entrenamiento.

    :param k: El parámetro 'k' representa la cantidad de vecinos más cercanos a
            considerar al hacer predicciones. En este caso,
            está configurado en 9.
    :param X_train: El parámetro `X_train` representa las características de
            los datos de entrenamiento, que son las variables de entrada
            utilizadas para entrenar el modelo KNN. Debe ser un DataFrame de
            pandas que contenga los datos de entrenamiento.
    :param y_train: El parámetro `y_train` representa la variable objetivo o
            las etiquetas de los datos de entrenamiento. Contiene las
            categorías reales para las filas correspondientes en `X_train`.
    :param X_test: La variable `X_test` representa el conjunto de datos de
            prueba, que contiene las características (variables independientes)
            para las cuales queremos predecir las categorías.
            Es un DataFrame de pandas.
    :return: La función `knn` devuelve una lista `assigned_categories` que
            contiene las categorías predichas para cada fila en `X_test`.
    """
    # Lista que contendrá todas las categorías asignadas para los
    # valores de X_train
    assigned_categories = []
    for _, row in X_test.iterrows():
        # Lo reinicia para cada nueva fila de X_test
        complete_row = np.array([])
        for _, test_point in row.items():
            # Para asegurarnos de no agregar valores None
            if test_point is not None:
                # Al terminar el ciclo 'for' 'complete_row' contendrá
                # los 9 valores de todas las columnas
                complete_row = np.append(complete_row, test_point)
        # Almacena las distancias para cada fila en X_train
        train_distances = []
        for index_train, row_train in X_train.iterrows():
            # Lo reinicia para cada nueva fila de X_train
            train_complete_row = np.array([])
            for _, train_point in row_train.items():
                if train_point is not None:
                    # Al terminar el ciclo 'for' 'train_complete_row'
                    # contendrá los 9 valores de todas las columnas
                    train_complete_row = np.append(train_complete_row,
                                                   train_point)
            # Calcula la distancia usando la función 'euclidean_distance'
            distance = euclidean_distance(complete_row, train_complete_row)
            # Esta lista contiene las distancias calculadas
            # con su respectiva categoría
            train_distances.append((distance, y_train.iloc[index_train]))
            # O debería considerar solo distancias no nulas y no cero?
            # if distance is not None and abs(distance) != 0.0:

        # Ordena la lista basandose en las distancias
        train_distances.sort(key=lambda x: x[0])
        # Solo usa las k distancias más cercanas
        nearests = train_distances[:k]

        # Calcula cuántos "2" y cuántos "4" hay en los vecinos más cercanos
        twos = sum(1 for _, label in nearests if label == 2)
        fours = sum(1 for _, label in nearests if label == 4)

        # Asigna la categoría basada en la mayoría de vecinos cercanos
        if twos > fours:
            assigned_categories.append(2)
        elif twos < fours:
            assigned_categories.append(4)

    # Compara las categorías predichas con las categorías reales
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
            """
            print("En la posición: ", i,
                  "predijo incorrectamente", predicted,
                  ", la categoría correcta era", actual)
            """

    print("\nEl porcentaje de predicciones correctas es de:",
          ((correct_predictions * 100)/len(assigned_categories)), "%")
    print("El porcentaje de predicciones incorrectas es de:",
          ((incorrect_predictions * 100)/len(assigned_categories)), "%")
    print("Número de predicciones correctas:", correct_predictions)
    print("Número de predicciones incorrectas:", incorrect_predictions)
    print("Índices de predicciones incorrectas:", misclassified_indices, "\n")

    return assigned_categories


closest_labels = knn(9, X_train, y_train, X_test)
print("Las", len(closest_labels), "variables: \n", X_test, "\n",
      "\n Se les predijo que sus categorías son: ")
print(closest_labels)
print("Estas son las categorías originales: \n", y_test.to_list())
