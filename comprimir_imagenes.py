import os
import cv2
import numpy as np

#ajustar las imagenes
pixels = (256, 256)
# Definir la carpeta que contiene las imágenes
carpeta = "TOGKush"

# Lista para almacenar las imágenes y etiquetas
imagenes = []
etiquetas = []

# Recorrer los archivos en la carpeta
for archivo in os.listdir(carpeta):
    # Leer la imagen
    ruta_imagen = os.path.join(carpeta, archivo)
    imagen = cv2.imread(ruta_imagen)
    #redimensionar la imagen
    try:
        imagen_redimensionada = cv2.resize(imagen, pixels)
        # Añadir la imagen a la lista de imágenes
        imagenes.append(imagen_redimensionada)
        # Añadir la etiqueta correspondiente
        etiquetas.append("OG Kush")
    except:
        #Si el formato de la imagen es incompatible lo ignoramos.
        pass
# Convertir listas a matrices NumPy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Guardar las matrices en un archivo comprimido
np.savez_compressed('TOGKush_images.npz', imagenes=imagenes, etiquetas=etiquetas)
