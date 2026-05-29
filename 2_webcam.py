import cv2
import numpy as np
import tensorflow as tf

print("cargando el modelo neuronal...")

# comentar o descomentar según el modelo que se quiera usar para la webcam
#modelo = tf.keras.models.load_model('modelo_definitivo.keras')
modelo = tf.keras.models.load_model('modelo_transfer_learning.keras')

clases = ['celulares', 'laptops', 'relojes']

cap = cv2.VideoCapture(1) # 0 es para la cámara x default, pero en mac me abría la de obs

print("cámara encendida. presiona la tecla 'q' en tu teclado para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("error al leer la cámara.")
        break

    # dimensiones del frame original
    alto, ancho, _ = frame.shape
    
    # lado del cuadrado = dimension más corta (usualmente el alto)
    lado = min(alto, ancho)
    
    # calculamos las coordenadas para centrar el cuadrado
    x_inicio = (ancho // 2) - (lado // 2)
    y_inicio = (alto // 2) - (lado // 2)
    
    # recorte de frame usando slicing de matrices en numpy
    frame_cuadrado = frame[y_inicio:y_inicio+lado, x_inicio:x_inicio+lado]
    # ----------------------------------------------

    # 1. preprocesamiento (ahora sobre el cuadrado perfecto, sin deformar la geometría)
    imagen_ia = cv2.resize(frame_cuadrado, (224, 224))
    imagen_ia = cv2.cvtColor(imagen_ia, cv2.COLOR_BGR2RGB)
    imagen_ia = np.expand_dims(imagen_ia, axis=0)

    # 2. inferencia
    prediccion = modelo.predict(imagen_ia, verbose=0)
    indice_clase = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100
    etiqueta = clases[indice_clase]

    # 3. visualización
    # ajustamos el rectángulo negro al ancho del nuevo frame cuadrado
    cv2.rectangle(frame_cuadrado, (0, 0), (lado, 40), (0, 0, 0), -1)
    
    texto = f"{etiqueta}: {confianza:.2f}%"
    color_texto = (0, 255, 0) if confianza > 70 else (0, 0, 255)
    
    # texto sobre el frame cuadrado
    cv2.putText(frame_cuadrado, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)

    # únicamente ventana recortada
    cv2.imshow('clasificador en tiempo real', frame_cuadrado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()