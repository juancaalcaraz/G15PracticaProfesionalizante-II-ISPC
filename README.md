![Texto alternativo](LOGOISPC.png)
## Materia: Práctica Profesionalizante II
**Carrera:** Ciencia de Datos e Inteligencia Artificial  
**Grupo:** 15  
**Cohorte:** 2022  

**Alumnos:**
- Juan Carlos Alcaraz - juan.alcaraz806@gmail.com
- Carla Contreras - karlaconty@gmail.com
- Natalia Cuestas - nataliacuestas32@hotmail.com
- Alan Lovera - alanroquegabriellovera@gmail.com
- Roberto Schiaffino - schiaffino35@gmail.com
- Eliana Karina Steinbrecher - karinasandua@gmail.com
- Sergio Tamietto - serdantam@gmail.com
## CannAlisys PRO
<div style="text-align: center;">
    <img src="Cannalisys.png" alt="CannAlisys" width="300">
</div>
## Mejora en la calidad del cannabis medicinal a través del procesamiento de imágenes

**Introducción:**
Durante las últimas décadas se ha establecido un creciente interés en la investigación y la elaboración de aceite de cannabis para el tratamiento de enfermedades y como paliativo del dolor. 
Sin embargo, para poder elaborar un aceite de calidad, es necesario el control de posibles enfermedades, de carencias de nutrientes, el control de plagas y la posible aparición de hongos. Garantizando así una calidad óptima de la planta a la hora de realizarse la extracción del aceite.

**Problema a abordar:**
El problema que se desea abordar es la detección temprana de factores que deterioran la salud del cannabis.

**Aplicación a explorar:**
Vamos a explorar cómo el procesamiento de imágenes junto a las redes neuronales profundas pueden ser una herramienta clave para este problema.

**Aspecto crucial del procesamiento de imágenes:**
La detección de patrones en las imágenes creemos que va a ser una herramienta fundamental del procesamiento de imágenes y de la I.A. para el cuidado de las plantas en el futuro de la agricultura.

## Resumen del Proyecto

La aplicación permite a los usuarios cargar imágenes de hojas de plantas, las cuales son procesadas por un modelo de aprendizaje profundo para identificar posibles enfermedades. La aplicación muestra la predicción del modelo y proporciona información relevante sobre la enfermedad identificada. También ofrece la opción de generar un archivo de audio con la información de la enfermedad para una accesibilidad mejorada.

## Librerías Utilizadas

- **numpy**: Para operaciones numéricas y manejo de arrays.
- **andas**: Para la manipulación y análisis de datos.
- **matplotlib**: Para la visualización de datos y gráficos.
- **opencv-python**: Para la manipulación y procesamiento de imágenes.
- **tensorflow**: Framework de aprendizaje profundo utilizado para construir y entrenar el modelo.
- **keras**: API de alto nivel de TensorFlow para la construcción y entrenamiento de modelos de redes neuronales.
- **scikit-learn**: Para dividir el conjunto de datos en entrenamiento, validación y prueba.
- **Flask**: Framework web para Python.
predicción.
- **gTTS**: Biblioteca para convertir texto a audio.
- **base64**: Biblioteca para codificar y decodificar datos en base64.

## Instalación

Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Coloca las imágenes de entrenamiento en el directorio `Train/`, organizadas en subdirectorios según sus categorías.
2. Ejecuta los script principales de entrenamiento:
    ```bash
    python Entrenamiento.py
    ```
3. El modelo entrenado se guardará como `full_model.h5`.
4. Ejecuta la aplicación Flask para la visualización web:
    ```bash
    python app.py
    ```

5. Abre tu navegador web y ve a `http://127.0.0.1:5000/` para acceder a la interfaz de usuario.
