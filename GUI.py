from importlib.resources import path
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image, ExifTags, ImageChops
from optparse import OptionParser
from datetime import datetime
import numpy as np
import random
import sys
import cv2
import re
import os
import itertools
from pyparsing import Opt
import io

import tensorflow as tf
from tensorflow.keras.preprocessing import image

from ForgeryDetection import Detect
import double_jpeg_compression
import copy_move_cfa


# Global variables
IMG_WIDTH = 400
IMG_HEIGHT = 400
uploaded_image = None

# copy-move parameters
cmd = OptionParser("usage: %prog image_file [options]")
cmd.add_option('', '--imauto',
               help='Automatically search identical regions. (default: %default)', default=1)
cmd.add_option('', '--imblev',
               help='Blur level for degrading image details. (default: %default)', default=8)
cmd.add_option('', '--impalred',
               help='Image palette reduction factor. (default: %default)', default=15)
cmd.add_option(
    '', '--rgsim', help='Region similarity threshold. (default: %default)', default=5)
cmd.add_option(
    '', '--rgsize', help='Region size threshold. (default: %default)', default=1.5)
cmd.add_option(
    '', '--blsim', help='Block similarity threshold. (default: %default)', default=200)
cmd.add_option('', '--blcoldev',
               help='Block color deviation threshold. (default: %default)', default=0.2)
cmd.add_option(
    '', '--blint', help='Block intersection threshold. (default: %default)', default=0.2)
opt, args = cmd.parse_args()
# if not args:
#     cmd.print_help()
#     sys.exit()


def getImage(path, width, height):
    
    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)

    return ImageTk.PhotoImage(img)


def browseFile():
    
    filename = filedialog.askopenfilename(title="Select an image", filetypes=[("image", ".jpeg"),("image", ".png"),("image", ".jpg")])

    # No file selected (User closes the browsing window)
    if filename == "":
        return

    global uploaded_image

    uploaded_image = filename

    progressBar['value'] = 0   # Reset the progress bar
    fileLabel.configure(text=filename)     # Set the path name in the fileLabel

    # Display the input image in imagePanel
    img = getImage(filename, IMG_WIDTH, IMG_HEIGHT)
    imagePanel.configure(image=img)
    imagePanel.image = img

    # Display blank image in resultPanel
    blank_img = getImage("images/output.png", IMG_WIDTH, IMG_HEIGHT)
    resultPanel.configure(image=blank_img)
    resultPanel.image = blank_img

    # Reset the resultLabel
    resultLabel.configure(text="READY TO SCAN", foreground="green")


def copy_move_forgery():
    # Retrieve the path of the image file
    path = uploaded_image
    eps = 60
    min_samples = 2

    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('Error', "Please select image")
        return

    detect = Detect(path)
    key_points, descriptors = detect.siftDetector()
    forgery = detect.locateForgery(eps, min_samples)

    # Set the progress bar to 100%
    progressBar['value'] = 100

    if forgery is None:
        # Retrieve the thumbs up image and display in resultPanel
        img = getImage("images/no_forgery.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Real Image", foreground="green")
    else:
        # Retrieve the output image and display in resultPanel
        img = getImage("images/copy_move.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Fake Image", foreground="red")
        cv2.imshow('Forgery', forgery)
        wait_time = 1000
        while(cv2.getWindowProperty('Forgery', 0) >= 0) or (cv2.getWindowProperty('Original image', 0) >= 0):
            keyCode = cv2.waitKey(wait_time)
            if (keyCode) == ord('q') or keyCode == ord('Q'):
                cv2.destroyAllWindows()
                break
            elif keyCode == ord('s') or keyCode == ord('S'):
                name = re.findall(r'(.+?)(\.[^.]*$|$)', path)
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                new_file_name = name[0][0]+'_'+str(eps)+'_'+str(min_samples)
                new_file_name = new_file_name+'_'+date+name[0][1]

                vaue = cv2.imwrite(new_file_name, forgery)
                print('Image Saved as....', new_file_name)


def cfa_artifact():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('Error', "Please select image")
        return

    identical_regions_cfa = copy_move_cfa.detect(path, opt, args)

    # Set the progress bar to 100%
    progressBar['value'] = 100

    if(identical_regions_cfa):
        # Retrieve the output image and display in resultPanel
        img = getImage("images/cfa.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text=f"{str(identical_regions_cfa)}, CFA artifacts detected", foreground="red")

    else:
        # print('\nSingle compressed')
        # Retrieve the thumbs up image and display in resultPanel
        img = getImage("images/no_cfa.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="NO-CFA artifacts detected", foreground="green")

def splicing():
    # Retrieve the path of the image file
    path = uploaded_image
    TEMP = 'temp.jpg'
    SCALE = 10

    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('Error', "Please select image")
        return

    original = Image.open(path)
    original.save(TEMP, quality=90)
    temporary = Image.open(TEMP)

    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    # Set the progress bar to 100%
    progressBar['value'] = 100
    diff.show()

    # Load the pretrained model
    model = tf.keras.models.load_model('model_casia_run1.h5')

    # Prepare the input data
    diff = diff.resize((128, 128))
    diff = np.array(diff)
    diff = np.expand_dims(diff, axis=0)
    diff = np.repeat(diff, 1, axis=-1)

    # Pass the input data through the model
    preds = model.predict(diff)

    # Interpret the output
    class_names = ['fake', 'real']
    predicted_class_idx = np.argmax(preds, axis = 1)[0] 
    predicted_class_label = class_names[predicted_class_idx]
    
    if predicted_class_label == 'real':
        # Retrieve the thumbs up image and display in resultPanel
        img = getImage("images/no_forgery.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Real Image", foreground="green")

    if predicted_class_label == 'fake':
        # Retrieve the output image and display in resultPanel
        img = getImage("images/splicing.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Fake Image", foreground="red")

def jpeg_Compression():

    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('Error', "Please select image")
        return

    double_compressed = double_jpeg_compression.detect(path)

    # Set the progress bar to 100%
    progressBar['value'] = 100

    if(double_compressed):
        # print('\nDouble compression detected')
        # Retrieve the output image and display in resultPanel
        img = getImage("images/double_compression.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Double compression", foreground="red")

    else:
        # print('\nSingle compressed')
        # Retrieve the thumbs up image and display in resultPanel
        img = getImage("images/single_compression.png", IMG_WIDTH, IMG_HEIGHT)
        resultPanel.configure(image=img)
        resultPanel.image = img

        # Display results in resultLabel
        resultLabel.configure(text="Single compression", foreground="green")


# Initialize the app window
root = tk.Tk()
root.title("Forgery Detector")
root.configure(bg="#F0FBFC")

# Ensure the program closes when window is closed
root.protocol("WM_DELETE_WINDOW", root.quit)

# Maximize the size of the window
root.state("zoomed")

# Label for the results of scan
resultLabel = Label(text="IMAGE FORGERY DETECTOR", font=("Times", 50), bg='#F0FBFC')
resultLabel.grid(row=0, column=0, columnspan=4)

# Get the blank image
input_img = getImage("images/input.png", IMG_WIDTH, IMG_HEIGHT)
middle_img = getImage("images/middle.png", IMG_WIDTH, IMG_HEIGHT)
output_img = getImage("images/output.png", IMG_WIDTH, IMG_HEIGHT)

# Displays the input image
imagePanel = Label(image=input_img, bg='#F0FBFC')
imagePanel.image = input_img
imagePanel.grid(row=1, column=0, padx=5)

# Label to display the middle image
middle = Label(image=middle_img, bg='#F0FBFC')
middle.image = middle_img
middle.grid(row=1, column=1, padx=5)

# Label to display the output image
resultPanel = Label(image=output_img, bg='#F0FBFC')
resultPanel.image = output_img
resultPanel.grid(row=1, column=2, padx=5)

# Label to display the path of the input image
fileLabel = Label(text="No file selected", font=("Times", 15), bg='#F0FBFC')
fileLabel.grid(row=2, column=1)
# fileLabel.grid(row=2, column=0, columnspan=2)


# Create the progress bar
progressBar = ttk.Progressbar(length=497)
progressBar.grid(row=3, column=1)

# Button to upload images
uploadButton = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15), text="Upload Image", width=44, command=browseFile)
uploadButton.grid(row=4, column=1, sticky="nsew", pady=5)

# Button to run the Copy-Move  detection algorithm
copy_move = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15), text="Copy-Move Detection", width=20, command=copy_move_forgery)
copy_move.grid(row=5, column=0, columnspan=2, pady=20)

# Button to run the Error-Level Analysis algorithm
ela = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15), text="Splicing Detection", width=20, command=splicing)
ela.grid(row=5, column=1, columnspan=2, pady=20)

# Button to run the Compression detection algorithm
compression = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15), text="Compression Detection", width=20, command=jpeg_Compression)
compression.grid(row=6, column=0, columnspan=2, pady=20)


# Button to run the CFA-artifact detection algorithm
artifact = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15), text="CFA-artifact Detection", width=20, command=cfa_artifact)
artifact.grid(row=6, column=1, columnspan=2, pady=20)


# Button to exit the program
quitButton = tk.Button(root, 
                bg='#fffd6e',
                fg='black',
                relief='solid',
                bd=2,
                highlightbackground='#6A340E',
                font=('Times', 15, 'bold'), text="Exit program", width=10, command=root.quit)
quitButton.grid(row=6, column=2, columnspan=2, pady=10)

# Open the GUI
root.mainloop()