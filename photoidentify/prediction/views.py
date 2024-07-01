from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.models import save_model # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.applications.vgg16 import decode_predictions # type: ignore

model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')

from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            top_preds = decode_predictions(result, top=5)[0]
            img_data = request.POST.get('img_data')
            for pred in top_preds:
                print("Predicted class: {}, Probability: {:.2f}%".format(pred[1], pred[2] * 100))
            return render(request, 'home.html', {'form': form, 'prediction': result, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
