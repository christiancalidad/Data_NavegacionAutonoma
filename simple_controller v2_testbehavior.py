
"""camera_pid controller."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importar las bibliotecas necesarias para el control del robot y procesamiento de imágenes
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
from time import sleep
import tensorflow as tf
import joblib
from PIL import Image

# Funcion para obtener la imagen de la cámara:
# Capturamos una imagen desde la cámara del robot y se convierte a un arreglo numpy.
def get_image(camera):
    #Obtener la imagen cruda de la camara
    raw_image = camera.getImage()
    #Se convierte la imagen cruda a un arreglo numpy
    image = np.frombuffer(raw_image, np.uint8)
    #Se crea una imagen PIL desde el arreglo numpy
    image_pil = Image.frombytes("L", (camera.getWidth(), camera.getHeight()), image)
    image_np = np.array(image_pil)
    print(image_np)
    #Retornar la imagen como arreglo numpy
    return image_np


# Función para mostrar la imagen en la pantalla:
# Muestra una imagen en la pantalla del robot.
def display_image(display, image):
    #Convertir la imagen a formato RGB
    image_rgb = np.dstack((image, image,image,))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    ) #Se crea una nueva imagen para mostrar
    #Pegar la imagen en la pantalla
    display.imagePaste(image_ref, 0, 0, False)

# Inicializacion de angulos y velocidad 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20

# Función para predecir el ángulo de dirección utilizando un modelo de aprendizaje profundo:
# Sen usa un modelo de aprendizaje profundo para predecir el ángulo de dirección basado en una imagen de entrada.
def predict_steering_angle(model, image):
    #Normalizar la imagen
    image = image.astype(np.float32) / 255  
    #Realizar la predicción
    prediction = model.predict(np.expand_dims(image, axis=-1))  
    #Retornar la predicción
    return prediction 

# Función para establecer la velocidad objetivo:
# Se establece la velocidad objetivo del robot.
def set_speed(kmh):
    global speed            #robot.step(50)
                            #update steering angle

# Función para actualizar el ángulo de dirección:
# Se actualiza el ángulo de dirección del robot.
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    #Establecer el ángulo de dirección
    steering_angle = wheel_angle
    #Actualizar el ángulo
    angle = wheel_angle

# Función para validar e incrementar el ángulo de dirección:
# Valida e incrementa el ángulo de dirección manualmente.
def change_steer_angle(inc):
    global manual_steering
    #Se aplica incremento
    new_manual_steering = manual_steering + inc
    #Se valida el intervalo
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        #Validar y establecer el nuevo ángulo de dirección
        manual_steering = new_manual_steering
        #Actualizar el ángulo de dirección
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:    
        pass
    else:
        turn = "left" if steering_angle < 0 else "right"


# Funcion principal del programa:
# Se inicia el proceso de control del robot basado en la predicción del ángulo de dirección mediante 
# aprendizaje profundo.
def main():

    global speed
    #Crear una instancia del robot
    robot = Car()
    #Crear una instancia del conductor
    driver = Driver()

    #Obtener el paso de tiempo del mundo actual
    timestep = int(robot.getBasicTimeStep())

    #Crear una instancia de la cámara
    camera = robot.getDevice("camera")
    #Habilitar la cámara con el paso de tiempo
    camera.enable(timestep)
    
    #Crear una instancia del radar
    radar = robot.getDevice("radar")
    #Habilitar el radar con el paso de tiempo
    radar.enable(timestep)
    
    #Crear una instancia del sensor de distancia
    distance_sensor = robot.getDevice("distance sensor")
    #Habilitar el sensor de distancia con el paso de tiempo
    distance_sensor.enable(timestep)
    
    #Definir umbrales y variables de control
    threshold_distance = 7      #Distancia umbral para detección de obstáculos
    brake_power = 1             #Potencia de frenado inicial
    reduce_speed = False        #Indicador para reducir la velocidad
    
    #Cargar el escalador
    scaler = joblib.load('scaler.pkl')
    #processing display
    #display_img = Display("display_image")
    
    #Cargar el modelo de aprendizaje profundo
    model = tf.keras.models.load_model('model_project_without_preprocessing.h5',compile=False)

    #Crear una instancia del teclado para capturar entradas del usuario
    keyboard=Keyboard()
    #Habilitar el teclado con el paso de tiempo
    keyboard.enable(timestep)
    
    #Definir la ruta para guardar las imágenes
    image_save_path = os.path.join(os.getcwd(), "test_images")
    #Definir la ruta para guardar el archivo CSV
    csv_file_path = os.path.join(os.getcwd(), "image_data_test.csv")
    last_file_name = ''

    #Crear el archivo CSV y escribir la cabecera si no existe
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Name", "Angle"])

    #Bucle principal de control del robot
    while robot.step() != -1:
        global speed
        #Obtener los objetivos detectados por el radar
        targets = radar.getTargets()
        should_stop = False   #Indicador para detener el vehículo

        #Evaluar cada objetivo detectado por el radar
        for target in targets:
            #Obtener la distancia del objetivo
            distance = target.distance
            #Verificar si el objetivo está dentro del umbral
            if distance <= threshold_distance and distance > 1:
                brake_power += 10    #Incrementar la potencia de frenado
                should_stop = True   #Indicar que se debe detener el vehículo
                print(f"Target detectado a distancia: {distance} metros")
                break  #Salir del bucle al encontrar un objetivo cercano
        
        #Obtener la distancia medida por el sensor de distancia
        distance = distance_sensor.getValue()
        #Verificar si la distancia medida está por debajo del umbral
        if distance < 1.5:
            should_stop = True  #Indicar que se debe detener el vehículo
            print(f"Obstáculo detectado por el sensor de distancia: {distance} metros")

        #Reducir la velocidad o detener el vehículo según los indicadores
        if should_stop:
            reduce_speed = True
            
        else:
            if reduce_speed:
                speed -= brake_power       #Reducir la velocidad gradualmente
                if speed <= 0:
                    speed = 0              #Asegurarse de que la velocidad no sea negativa
                    reduce_speed = False
            else:
                if speed < 20:
                    speed += 1 
        
        print(speed)


        #driver.setCruisingSpeed(speed)

        #Obtener la imagen de la cámara
        image = get_image(camera)
        predicted_angle = predict_steering_angle(model, image)
        predicted_angle = scaler.inverse_transform(predicted_angle)
        global angle
        angle = float(predicted_angle[[0]])  #Actualizar el ángulo con la predicción
        print(speed,angle)
        #Capturar la entrada del teclado
        current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f"))
        file_name = current_datetime + ".png"

        key = keyboard.getKey()
        if key == keyboard.UP:
            global wheel_angle
            global manual_steering
            angle = 0
            wheel_angle = 0
            manual_steering = 0
            print("up")
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT:
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            pass
        
        #Establecer el ángulo de dirección y la velocidad de crucero del conductor
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

# Ejecutar la función principal si el archivo se ejecuta directamente
if __name__ == "__main__":
    main()