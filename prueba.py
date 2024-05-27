import cv2
import numpy as np
from keras.models import model_from_json

with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Crear el modelo a partir de la arquitectura cargada
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos del modelo
loaded_model.load_weights("model.weights.h5")

print("Modelo cargado correctamente.")

# Preprocesamiento de la imagen
def preprocess_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    # Asegurarse de que la imagen tiene el mismo tamaño que durante el entrenamiento
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    # Normalizar los valores de píxeles
    image = image / 255.0
    # Expandir las dimensiones para que coincidan con el formato de entrada del modelo
    image = np.expand_dims(image, axis=0)
    return image

# Hacer predicciones sobre una imagen
def predict_image(image_path, model):
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)
    # Hacer la predicción utilizando el modelo cargado
    prediction = model.predict(preprocessed_image)
    # Obtener el índice de la clase con la probabilidad más alta
    predicted_class_index = np.argmax(prediction)
    # Suponiendo que tienes un diccionario que mapea los índices de clase a nombres de clase
    class_names = {0: "Deficiencia de nitrogeno", 1: "Hojas libres de enfermedades", 2: "Hongos de oxido",3: "Hojas libres de enfermedades", 4: "Infestacion de mosca blanca", 5: "Infestacion de orugas", 6: "Infestacion de pulgones", 7: "Marchitez por fusarium", 8: "Moho polvoriento", 9: "Podredumbre negra"}  # Reemplaza con tus nombres de clase
    # Obtener el nombre de la clase predicha
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Ruta de la imagen que quieres analizar
image_path = "images (2).jpg"

# Realizar la predicción
predicted_class = predict_image(image_path, loaded_model)

print("La imagen pertenece a la clase:", predicted_class)