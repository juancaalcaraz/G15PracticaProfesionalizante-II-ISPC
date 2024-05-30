import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json

# Definimos el tamaño de las imágenes
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Ruta al dataset
dataset_path = "Train"

# Listas para almacenar las imágenes y etiquetas
images = []
labels = []

# Cargar imágenes y etiquetas
print("Cargando imágenes...")
total_images = 0
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        
        # Verificar si el archivo es una imagen (JPEG o PNG)
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            image = cv2.imread(image_path)
            if image is not None:
                total_images += 1
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
                images.append(image)
                labels.append(label)
            else:
                print(f"Error al cargar la imagen: {image_path}")
        else:
            print(f"Ignorando archivo no válido: {image_file}")

print(f"Total de imágenes cargadas: {total_images}")

# Convertir listas a arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Normalizar imágenes
images = images / 255.0

# Codificar etiquetas
label_to_id = {label: idx for idx, label in enumerate(np.unique(labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
labels = np.array([label_to_id[label] for label in labels])
labels = to_categorical(labels)

#from sklearn.model_selection import train_test_split
# División del Dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Construcción del Modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_id), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento del Modelo
print("Entrenando el modelo...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(X_val, y_val))

# Imprimir métricas de entrenamiento y validación después de cada época
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch + 1}/{len(history.history['accuracy'])}")
    print(f"  - Precisión de entrenamiento: {history.history['accuracy'][epoch]}")
    print(f"  - Pérdida de entrenamiento: {history.history['loss'][epoch]}")
    print(f"  - Precisión de validación: {history.history['val_accuracy'][epoch]}")
    print(f"  - Pérdida de validación: {history.history['val_loss'][epoch]}")

# Evaluación del Modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Implementación del Modelo
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = id_to_label[np.argmax(prediction)]
    return predicted_label

# Ejemplo de uso
print(predict_image("TEST.JPG"))

model.save("full_model.h5")