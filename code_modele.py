
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Nom des classes
classes = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Baskets', 'Sac', 'Bottines'
]

# Chargement des données
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Création du modèle avec perceptrons
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # Perceptron 1
    Dense(96, activation='relu'),   # Perceptron 2
    Dense(64, activation='relu'),   # Perceptron 3
    Dense(48, activation='relu'),   # Perceptron 4
    Dense(32, activation='relu'),   # Perceptron 5
    Dense(10, activation='softmax') # Perceptron 6
])

# Compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement
model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)

# Évaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print("Précision sur les données de test :", test_accuracy)

# Fonction de prédiction interactive
def predire_image():
    while True:
        entree = input("Entrez un numéro d'image entre 0 et 9999 (ou 'exit' pour quitter) : ")
        if entree.lower() == 'exit':
            break
        try:
            index = int(entree)
            if 0 <= index < len(x_test):
                image = x_test[index]
                prediction = model.predict(np.expand_dims(image, axis=0))
                classe_predite = np.argmax(prediction)
                nom_classe = classes[classe_predite]

                print(f"Classe prédite : {nom_classe} ({classe_predite})")

                plt.imshow(image, cmap='gray')
                plt.title(f"Classe prédite : {nom_classe}")
                plt.axis('off')
                plt.show()
            else:
                print("Veuillez entrer un nombre entre 0 et 9999.")
        except ValueError:
            print("Entrée invalide. Entrez un nombre ou 'exit'.")

# Lancer la prédiction
predire_image()
