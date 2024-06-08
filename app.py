from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:  # Check if the image is loaded successfully
                image = cv2.resize(image, (100, 100))  # Resize the image to a fixed size
                images.append(image)
                labels.append(i)
    return np.array(images), np.array(labels)

# Prepare the dataset
data_dir = "C:/Users/prati/OneDrive/Desktop/Medical_project/preprocessed"
images, labels = load_data(data_dir)

# Flatten the images
images_flat = images.reshape(images.shape[0], -1)

# Train the SVM model
svm_model = SVC(kernel='linear', probability=True)  # Set probability=True for predict_proba
svm_model.fit(images_flat, labels)

# Save the model
joblib.dump(svm_model, 'svm_model.pkl')

def predict_image(img_path):
    # Load the saved model
    loaded_model = joblib.load('svm_model.pkl')
    
    # Load and preprocess the new input image
    new_image = cv2.imread(img_path)
    if new_image is None:
        return "unrecognized"
    
    new_image = cv2.resize(new_image, (100, 100))  # Resize the image to match training data size
    new_image_flat = new_image.reshape(1, -1)
    
    # Predict the class of the new input image
    predicted_proba = loaded_model.predict_proba(new_image_flat)
    print("Predicted Probabilities:", predicted_proba)
    
    max_proba = np.max(predicted_proba)
    print("Max Probability:", max_proba)
    
    # Set a threshold for confidence
    threshold = 0.5
    
    if max_proba < threshold:
        return "unrecognized"
    
    predicted_class = np.argmax(predicted_proba)
    print("Predicted Class Index:", predicted_class)
    
    # Get the name of the predicted class
    class_names = sorted(os.listdir(data_dir))
    predicted_class_name = class_names[int(predicted_class)]
    print("Predicted Class Name:", predicted_class_name)

    return predicted_class_name

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_path = "temp_img.jpg"
        img.save(img_path)
        predicted_class = predict_image(img_path)
        return jsonify({'predicted_class': predicted_class})

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        keyword = request.form['keyword']
        if keyword.lower() == 'alovera':
            return redirect(url_for('alovera'))
        elif keyword.lower() == 'adulasa':
            return redirect(url_for('adulasa'))
        elif keyword.lower() == 'giloy':
            return redirect(url_for('giloy'))
        elif keyword.lower() == 'guava':
            return redirect(url_for('guava'))
        elif keyword.lower() == 'neem':
            return redirect(url_for('neem'))
        elif keyword.lower() == 'tulasi':
            return redirect(url_for('tulasi'))
        else:
            # Redirect to another page or handle other cases
            pass
    # Render default page if no keyword or keyword doesn't match
    return render_template('index2.html')

@app.route('/alovera')
def alovera():
    return render_template('alovera.html')

@app.route('/adulasa')
def adulasa():
    return render_template('adulsa.html')

@app.route('/giloy')
def giloy():
    return render_template('giloy.html')

@app.route('/guava')
def guava():
    return render_template('guava.html')

@app.route('/neem')
def neem():
    return render_template('neem.html')

@app.route('/tulasi')
def tulasi():
    return render_template('tulasi.html')



if __name__ == '__main__':
    app.run(debug=True)
