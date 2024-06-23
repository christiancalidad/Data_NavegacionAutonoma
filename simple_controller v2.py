"""camera_pid controller."""

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
#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage() 
    image = np.frombuffer(raw_image, np.uint8)
    image_pil = Image.frombytes("RGBA", (camera.getWidth(), camera.getHeight()), image)
    image_np = np.array(image_pil)
    return image_np


#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20


def predict_steering_angle(model, image):
    image = image.astype(np.float32) /255
    prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    steering_angle = wheel_angle

    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:    
        pass
    else:
        turn = "left" if steering_angle < 0 else "right"


# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    scaler = joblib.load('scaler.pkl')
    # processing display
    #display_img = Display("display_image")
    model = tf.keras.models.load_model('model_project_without_preprocessing.h5',compile=False)

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)
    image_save_path = os.path.join(os.getcwd(), "test_images")
    csv_file_path = os.path.join(os.getcwd(), "image_data_test.csv")
    last_file_name = ''

    # Create the CSV file and write the header if it doesn't exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Name", "Angle"])

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)
        predicted_angle = predict_steering_angle(model, image)
        predicted_angle = scaler.inverse_transform(predicted_angle)
        print(predicted_angle)
        global angle
        angle = float(predicted_angle[0])
        # # Process and display image 
        # grey_image = greyscale_cv2(image)
        # display_image(display_img, grey_image)
        # # Read keyboard
        current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f"))
        file_name = current_datetime + ".png"
        
        print(os.getcwd() + "/" + file_name)
        print(angle)
        #image_pil = Image.fromarray(image)
        #image_pil.save(os.path.join(image_save_path, file_name))
            
        
            
        # #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()