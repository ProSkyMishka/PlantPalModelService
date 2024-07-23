import io
import json

from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
import os
from joblib import dump, load
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

with open('plantnet300K_species_names.json', 'r') as f:
    species_names = json.load(f)

val_dir = 'ModelsTraining/plantnet_300K/images_val'

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    'val': datasets.ImageFolder(val_dir, data_transforms['val']),
}

model_plant = models.resnet50(weights='DEFAULT')
num_ftrs = model_plant.fc.in_features
model_plant.fc = nn.Linear(num_ftrs, 102)
model_plant.load_state_dict(torch.load('model_best_accuracy.pth', map_location="cpu"))
model_plant.eval()


model_disease = models.resnet50(weights='DEFAULT')
num_disease_ftrs = model_disease.fc.in_features
model_disease.fc = nn.Linear(num_disease_ftrs, 6)
model_disease.load_state_dict(torch.load('disease_model_best_accuracy.pth', map_location="cpu"))
model_disease.eval()

def preprocess_disease_image(image):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

disease_names = ["leaf spot",
            "calcium deficiency",
            "leaf scorch",
            "leaf blight",
            "curly yellow virus",
            "yellow vein mosaic"]

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        print("Изображение не найдено в запросе")
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    print("Изображение получено")

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except IOError:
        return jsonify({'error': 'Invalid image format'}), 400

    image = preprocess_disease_image(image)

    with torch.no_grad():
        outputs = model_disease(image)
        print(outputs[0])
        a, preds = torch.max(outputs, 1)
        print(a, preds)
        predicted_disease_id = preds.item()
        predicted_disease = disease_names[predicted_disease_id]
        print(predicted_disease)

    response = f"{outputs[0]}"
    return jsonify({'response': f"{response[8:(len(response) - 2)]}"})


@app.route('/predict/plant', methods=['POST'])
def predict():
    print("Получен запрос")

    print(request)
    if 'image' not in request.files:
        print("Изображение не найдено в запросе")
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    print("Изображение получено")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        print("Изображение открыто")
    except IOError:
        print("Неверный формат изображения")
        return jsonify({'error': 'Invalid image format'}), 400

    image = preprocess_image(image)
    print("Изображение предобработано")

    with torch.no_grad():
        outputs = model_plant(image)
        print(outputs[0])
        print(len(outputs[0]))
        a, preds = torch.max(outputs, 1)
        print(a)
        response = f"{a}"
        print(preds)
        predicted_class = preds.item()
        print('Predicted class:', predicted_class)
        predicted = image_datasets['val'].classes[predicted_class]
        print(predicted)
        real_name = species_names[predicted]
        print(real_name)
        print("Предсказание выполнено")

    print("Отправка ответа")
    return jsonify({'classML': predicted, 'real_name': real_name, 'accuracy': f"{response[8:(len(response) - 2)]}"})

if __name__ == '__main__':
    app.run(host ="0.0.0.0", debug=True, port=8000)
