import cv2
import numpy as np
import tensorflow as tf

print("cargando el modelo neuronal...")
modelo = tf.keras.models.load_model('modelo_definitivo.keras')

clases = ['celulares', 'laptops', 'relojes']

cap = cv2.VideoCapture(1) # 0 es para la cámara x default, pero en mi mac me abría la de obs

print("cámara encendida. presiona la tecla 'q' en tu teclado para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("error al leer la cámara.")
        break

    # 1. preprocesamiento
    # copia reducida a 224x224
    imagen_ia = cv2.resize(frame, (224, 224))
    
    # opencv captura en formato bgr y habíamos entrenado con rgb
    imagen_ia = cv2.cvtColor(imagen_ia, cv2.COLOR_BGR2RGB)
    
    # convertimos la imagen a un tensor agregando la dimensión del lote (batch)
    # (224, 224, 3) a (1, 224, 224, 3)
    imagen_ia = np.expand_dims(imagen_ia, axis=0)

    # 2. inferencia
    prediccion = modelo.predict(imagen_ia, verbose=0)
    
    # extraemos el índice con el valor más alto y su porcentaje de seguridad
    indice_clase = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100
    etiqueta = clases[indice_clase]

    # 3. visualización en pantalla
    # rectángulo negro semitransparente arriba para que el texto se lea bien
    cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
    
    texto = f"{etiqueta}: {confianza:.2f}%"
    
    # > 70% segura, texto verde
    # si duda, texto rojo
    color_texto = (0, 255, 0) if confianza > 70 else (0, 0, 255)
    
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)

    cv2.imshow('clasificador en tiempo real', frame)

    # 4. condición de salida
    # 'q' rompe el bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# limpieza de memoria y apagado del hardware
cap.release()
cv2.destroyAllWindows()