from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import tensorflow as tf
import base64
from gtts import gTTS
import os

app = Flask(__name__)

# Datos de las categorías
data = {
    "Deficiencia de nitrogeno": {
        "Identificacion": "Coloración amarilla en hojas más viejas",
        "Daño": "Hojas pálidas o amarillas, crecimiento lento, retraso en la floración",
        "Causa": "Deficiencia de nitrógeno en el suelo, mal equilibrio de pH",
        "Prevencion": "Fertilización con abonos ricos en nitrógeno, ajuste del pH del suelo, rotación de cultivos para mejorar la salud del suelo"
    },
    "Hojas libres de enfermedades": {
        "Identificacion": "Coloración verde uniforme, sin manchas ni deformaciones",
        "Daño": "Ninguno",
        "Causa": "No aplica",
        "Prevencion": "Mantenimiento adecuado de la planta, control de plagas y enfermedades, prácticas agrícolas higiénicas"
    },
    "Hongos de oxido": {
        "Identificacion": "Manchas de color marrón o anaranjado en las hojas",
        "Daño": "Manchas de óxido en las hojas, deformación y caída prematura de las hojas",
        "Causa": "Condiciones de alta humedad, falta de ventilación",
        "Prevencion": "Aireación adecuada, eliminación de hojas infectadas, tratamiento preventivo con fungicidas"
    },
    "Infestacion de acaros": {
        "Identificacion": "Presencia de telarañas finas y manchas blancas en las hojas",
        "Daño": "Hojas enrolladas, manchas blancas en el envés de las hojas, defoliación",
        "Causa": "Ambientes secos y calurosos, falta de humedad relativa en el aire",
        "Prevencion": "Aumento de la humedad ambiental, enemigos naturales como ácaros depredadores, aplicación de insecticidas específicos"
    },
    "Infestacion de mosca blanca": {
        "Identificacion": "Pequeños insectos blancos voladores sobre la planta",
        "Daño": "Hojas amarillas y pegajosas, daño en el follaje, transmisión de virus",
        "Causa": "Condiciones cálidas y húmedas, plantas débiles o estresadas",
        "Prevencion": "Control biológico con depredadores naturales, trampas pegajosas, aplicación de insecticidas específicos"
    },
    "Infestacion de orugas": {
        "Identificacion": "Presencia de orugas y daño en las hojas",
        "Daño": "Hojas mordisqueadas, agujeros en el follaje, excrementos de orugas en las hojas",
        "Causa": "Orugas en estado larval que se alimentan de las hojas de la planta",
        "Prevencion": "Inspección regular de la planta, eliminación manual de orugas, uso de repelentes naturales"
    },
    "Infestacion de pulgones": {
        "Identificacion": "Pequeños insectos de color verde o negro en las hojas",
        "Daño": "Hojas enrolladas o deformadas, secreción pegajosa en las hojas, crecimiento retardado de la planta",
        "Causa": "Condiciones cálidas y secas, plantas débiles o estresadas",
        "Prevencion": "Control biológico con depredadores naturales, enjuague con agua jabonosa, aplicación de insecticidas específicos"
    },
    "Marchitez por fusarium": {
        "Identificacion": "Marchitamiento repentino de las hojas y los tallos",
        "Daño": "Marchitamiento de hojas y tallos, oscurecimiento de las raíces, disminución del crecimiento de la planta",
        "Causa": "Infección fúngica por Fusarium en el suelo",
        "Prevencion": "Uso de semillas y sustratos libres de Fusarium, rotación de cultivos, prácticas de desinfección del suelo"
    },
    "Moho polvoriento": {
        "Identificacion": "Capa blanca polvorienta en las hojas y los tallos",
        "Daño": "Manchas blancas polvorientas en las hojas, deformación del follaje, detención del crecimiento de la planta",
        "Causa": "Condiciones de alta humedad y baja circulación de aire, plantas estresadas por falta de luz o nutrientes",
        "Prevencion": "Mejora de la circulación de aire, reducción de la humedad, aplicación de fungicidas, eliminación de hojas infectadas"
    },
    "Podredumbre negra": {
        "Identificacion": "Manchas húmedas y oscuras en hojas, tallos y frutos",
        "Daño": "Manchas húmedas y oscuras en las hojas y los tallos, pudrición de las raíces, caída prematura de frutos",
        "Causa": "Infección bacteriana o fúngica en las plantas debido a condiciones de alta humedad y mala ventilación",
        "Prevencion": "Mejora de la circulación de aire, manejo adecuado del riego, aplicación de fungicidas y bactericidas, eliminación de material vegetal infectado"
    }
}

# Cargar el modelo completo
loaded_model = tf.keras.models.load_model("Models/full_model.h5")
print("Modelo cargado correctamente.")

# Preprocesamiento de la imagen
def preprocess_image(image):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Hacer predicciones sobre una imagen
def predict_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    class_names = {0: "Deficiencia de nitrogeno", 1: "Hojas libres de enfermedades", 2: "Hongos de oxido",
                   3: "Infestacion de acaros", 4: "Infestacion de mosca blanca", 5: "Infestacion de orugas",
                   6: "Infestacion de pulgones", 7: "Marchitez por fusarium", 8: "Moho polvoriento",
                   9: "Podredumbre negra"}
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Cargar el modelo MobileNetSSD para detección de plantas
net = cv2.dnn.readNetFromCaffe("Models/deploy.prototxt", "Models/mobilenet_iter_73000.caffemodel")

# Clases de MobileNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_plants(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    plant_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "pottedplant":
                plant_detected = True
                break

    if plant_detected:
        return image, True
    else:
        return None, False

# Ruta de la página principal
@app.route('/')
def index():
    return render_template('index.html', prediction_result=None, image_data=None)

# Ruta para manejar la carga de imágenes y realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detectar plantas en la imagen
    detected_plants_image, plant_detected = detect_plants(image)

    if not plant_detected:
        prediction_result = "No se detectaron plantas. Intente cargar otra imagen."
        image_data = None
        result_text = "No se detectaron plantas. Intente cargar otra imagen."
    else:
        # Convertir la imagen con detecciones a base64 para mostrarla en HTML
        _, img_encoded = cv2.imencode('.jpg', image)
        image_data = base64.b64encode(img_encoded).decode('utf-8')

        # Realizar la predicción de la enfermedad
        predicted_class = predict_image(image, loaded_model)

        # Generar el texto completo para la síntesis de voz
        if predicted_class in data:
            result_text = f"Resultado: {predicted_class}. Identificación: {data[predicted_class]['Identificacion']}. Daño: {data[predicted_class]['Daño']}. Causa: {data[predicted_class]['Causa']}. Prevención: {data[predicted_class]['Prevencion']}."
        else:
            result_text = f"Resultado: {predicted_class}. No se encontraron detalles para esta categoría."

        prediction_result = predicted_class

    return render_template('index.html', prediction_result=prediction_result, image_data=image_data, data=data, result_text=result_text)

# Ruta para generar y enviar el archivo de audio
@app.route('/speak_result/<result_text>')
def speak_result(result_text):
    tts = gTTS(text=result_text, lang='es')
    tts.save('result.mp3')
    return send_file('result.mp3', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
