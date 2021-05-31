# ---------------------------------- Importing Libraries ------------------------------------------------#
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import shutil
import os
from PIL import Image, ImageTk
import cv2
import numpy as np

# ----------------------------- Disease Dictionary -------------------------------------------------------
disease_data = {0: 'Apple-Apple Scab', 1: 'Apple-Black Rot', 2: 'Apple-Cedar Apple Rust', 3: 'Apple-Healthy',
                4: 'Blueberry-Healthy', 5: 'Cherry-Powdery Mildew', 6: 'Cherry-Healthy',
                7: 'Corn-Cercospora Leaf Spot', 8: 'Corn-Common RS Rust', 9: 'Corn-Northern Leaf Blight', 10: 'Corn-Healthy',
                11: 'Grape-Black Rot', 12: 'Grape-Black Measles', 13: 'Grape-Leaf Blight', 14: 'Grape-Healthy',
                15: 'Orange-Haunglongbing(Citrus Greening)', 16: 'Peach-Bacterial Spot', 17: 'Peach-Healthy',
                18: 'Pepper-Bacterial Spot', 19: 'Pepper-Healthy', 20: 'Potato-Early Blight', 21: 'Potato-Late Blight',
                22: 'Potato-Healthy', 23: 'Raspberry-Healthy', 24: 'Soybean-Healthy', 25: 'Squash-Powdery Mildew',
                26: 'Strawberry-Leaf Scorch', 27: 'Strawberry-Healthy', 28: 'Tomato-Bacterial Spot',
                29: 'Tomato-Early Blight', 30: 'Tomato-Late Blight', 31: 'Tomato-Leaf Mold', 32: 'Tomato-Septoria Leaf Spot',
                33: 'Tomato-Spider Mites', 34: 'Tomato-Target Spot', 35: 'Tomato-Yellow Lead Curl Virus',
                36: 'Tomato-Mosaic Virus', 37: 'Tomato-Healthy'}

# ------------------------------- Function to analyze image ------------------------------------------------


def analysis():
    # Directory containing image to predict
    verify_dir = 'testpicture'

    # ------------------------ Function for reading image and prepare for prediction ----------------------
    def process_verify_data():
        for img in os.listdir(verify_dir):
            path = os.path.join(verify_dir, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img/255
            return img

    processed_img = process_verify_data()

    # Loading model and predicting image using model
    model = load_model('./model/leafDiseaseDetect.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    model_out = model.predict([processed_img])[0]
    group = np.argmax(model_out)

    # Getting Disease Name from Disease Dictionary using predicted class value
    str_label = disease_data[group]
    if str_label.__contains__('Healthy'):
        status = "HEALTHY"
    else:
        status = "UNHEALTHY"

    # Removing image from directory
    for f in os.listdir(verify_dir):
        os.remove(os.path.join(verify_dir, f))

    # Displaying status of leaf as Healthy or Unhealthy
    message = tk.Label(text='Status: '+status, background="lightgreen", fg="Red", font=("", 15, "bold"))
    message.grid(column=0, row=3, padx=10, pady=10)

    # ------------------------ Function to clear window for getting new input --------------------------------
    def again():
        if disease:
            disease.destroy()
        if r:
            r.destroy()
        if message:
            message.destroy()

        openphoto()

    # Displaying Disease Name, if leaf is unhealthy
    if status == 'UNHEALTHY':
        disease = tk.Label(text='Disease Name: ' + str_label, background="lawn green", fg="dark violet", font=("", 15))
        disease.grid(column=0, row=5, padx=10, pady=10)
        r = tk.Label(text='Click below to test new picture', background="lawn green", fg="brown4", font=("", 15))
        r.grid(column=0, row=6, padx=10, pady=10)
        button3 = tk.Button(relief='groove', activeforeground='Green', font=("Ariel", 10, "bold"), text="Check New", command=again)
        button3.grid(column=0, row=7, padx=10, pady=10)
    else:
        disease = tk.Label(text=str_label, background="lightgreen", fg="dark violet", font=("", 15))
        disease.grid(column=0, row=5, padx=10, pady=10)
        r = tk.Label(text='Click below to test new picture', background="lawn green", fg="brown4", font=("", 15))
        r.grid(column=0, row=6, padx=10, pady=10)
        button = tk.Button(relief='groove', activeforeground='Green', font=("Ariel", 10, "bold"), text="Check New", command=again)
        button.grid(column=0, row=7, padx=20, pady=20)

# ------------------------------- Function to ask for selecting image to test for disease ----------------------


def openphoto():
    dirpath = "./"
    filename = askopenfilename(initialdir=dirpath, title='Select image for analysis ', filetypes=[('image files', '.jpg')])
    dst = "./testpicture"
    shutil.copy(filename, dst)
    load = Image.open(filename)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=3, padx=10, pady=10)
    title.destroy()
    img_label.destroy()
    button1.destroy()
    button2 = tk.Button(relief='groove', activeforeground='Green', font=("Ariel", 10, "bold"), text="Analyse Image", command=analysis)
    button2.grid(column=0, row=5, padx=10, pady=10)


# -------------------------------------- Creating Window -------------------------------------------------------
window = tk.Tk()
window.title("Dr. Plant")
window.resizable(False, False)
window.geometry("500x510")
window.configure(background="lightgreen")

main_title = tk.Label(window, text="LEAF DISEASE PREDICTION", background="lawn green", fg="Red", font=("Comic Sans MS", 25, "bold"))
main_title.grid()

image1 = Image.open("./logo/logo.jpg")
image1 = image1.resize((450, 300), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)

img_label = tk.Label(image=test)
img_label.grid(pady=10)

title = tk.Label(window, text="Click below to choose picture for testing disease", background="lawn green", fg="brown4", font=("", 15))
title.grid(row=7, rowspan=3, pady=10)

button1 = tk.Button(window, relief='groove', activeforeground='Green', font=("Ariel", 10, "bold"), text="Get Photo", command=openphoto)
button1.grid(row=10, rowspan=3, padx=10, pady=10)
window.mainloop()