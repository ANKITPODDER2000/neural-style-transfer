from flask import Flask , render_template , request , url_for
# from flask_ngrok import run_with_ngrok
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import os

app = Flask(__name__)
# run_with_ngrok(app)


upload_path = "static/upload"
# print("**********************Start loading the model**********************")
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# print("**********************Model loading is done**********************")

def load_img(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return image[tf.newaxis , ]

def style_transfer(img1_path , img2_path):
    my_img = load_img(img1_path)
    style_img = load_img(img2_path)
    new_img = model(tf.constant(my_img) , tf.constant(style_img))
    new_img_get = np.array(new_img[0][0] * 255 , dtype=np.uint8)
    
    return new_img_get

@app.route("/" , methods = ['POST' , "GET"])
def home():
    # Get images from html form
    if request.method == "POST":
        if 'file1' not in request.files or 'file2' not in request.files:
            print('No file part')
            return render_template("home.html")
        file1 = request.files['file1']
        file2 = request.files['file2']
        filename1 = file1.filename
        filename2 = file2.filename
        filename1 = os.path.join(upload_path , filename1)
        filename2 = os.path.join(upload_path , filename2)
        file1.save(filename1)
        file2.save(filename2)

        style_img = style_transfer(filename1,filename2)
        img_name = "upload_img"+str(time.time()).replace(".","_")+".jpg"
        new_img_path = os.path.join(upload_path , img_name)
        try:
            os.remove(new_img_path)
        except:
            pass
        tf.keras.preprocessing.image.save_img(new_img_path , style_img)
        print([
                [filename1 , "Image 1"],
                [filename2 , "Image 2"],
                [new_img_path , "Resultant Image"]
            ])
        return render_template(
            "home.html" ,
            show_result = True ,
            image = [
                [filename1 , "Image 1"],
                [filename2 , "Image 2"],
                [new_img_path , "Resultant Image"]
            ]
        )
    return render_template("home.html" , show_result = False)

if __name__=="__main__":
    app.run(debug=True)