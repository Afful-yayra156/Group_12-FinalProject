# Group_12-FinalProject
Introduction to Artificial Intelligence Final Project. This project focuses on training a model to classify Nigerian foods and predict additional information such as food name, origin, health benefits and others when given an image.
Nigerian food Classification
Group12_Intro_to_AI
2024-08-05
Nigerian Foods and Snacks Image Classifier
This project aims to classify images of Nigerian foods and snacks using the InceptionV3 architecture. The classifier provides predictions along with additional nutritional and geographical information about the food.
Table of Contents
•	Overview
•	Dataset
•	Setup
•	Data Preprocessing
•	Model Training
•	Model Evaluation
•	Confusion Matrix
•	Additional Information Integration
•	Deployment
•	Usage
•	Contributing
Overview
The project leverages the InceptionV3 model to classify Nigerian foods and snacks into various categories. The model is fine-tuned to provide high accuracy and includes additional features like data augmentation and class weight adjustments to handle imbalanced data.
Dataset
The dataset used in this project can be downloaded from Kaggle. It contains images of various Nigerian foods and snacks divided into training, validation, and test sets.
Setup
1.	Clone the repository:
 	git clone https://github.com/yourusername/nigerian-foods-classifier.git
cd nigerian-foods-classifier
2.	Install the required packages:
 	pip install -r requirements.txt
3.	Download the dataset:
 	Make sure you have a Kaggle account and have your Kaggle API key set up.
 	!kaggle datasets download -d peaceedogun/nigerian-foods-and-snacks-multiclass
4.	Extract the dataset:
 	import zipfile
from pathlib import Path

zip_file_path = Path("C:/Users/user/OneDrive - Ashesi University/intro to ai/nigerian-foods-and-snacks-multiclass.zip")
extract_dir = Path("C:/Users/user/OneDrive - Ashesi University/intro to ai/nigerian-foods-and-snacks")

extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
5.	Set up paths to dataset:
 	train_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\train"
test_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\test"
val_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\validation"
whole_data = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass"
Data Preprocessing
1.	Check and remove corrupted images:
 	from PIL import Image
import os

def check_and_remove_corrupt_images(directory):
    corrupt_images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except (IOError, SyntaxError, OSError) as e:
                print(f'Removing bad file: {filepath}')
                corrupt_images.append(filepath)
                os.remove(filepath)
    return corrupt_images

whole_data = 'C:/Users/user/OneDrive - Ashesi University/intro to ai/nigerian-foods-and-snacks-multiclass'
corrupt_images = check_and_remove_corrupt_images(whole_data)
print(f'Found and removed {len(corrupt_images)} corrupt images.')
2.	Load and preprocess data:
 	from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_fp, test_fp, val_fp):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        train_fp,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )
    validation_gen = val_datagen.flow_from_directory(
        val_fp,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )
    test_gen = val_datagen.flow_from_directory(
        test_fp,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )
    return train_gen, validation_gen, test_gen

train_gen, validation_gen, test_gen = load_data(train_dir, test_dir, val_dir)
print(f"Number of training samples: {train_gen.samples}")
print(f"Number of validation samples: {validation_gen.samples}")
3.	Visualize class distribution:
 	import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(generator):
    class_counts = np.bincount(generator.classes)
    class_names = list(generator.class_indices.keys())
    sns.barplot(x=class_names, y=class_counts)
    plt.xticks(rotation=90)
    plt.title("Class Distribution")
    plt.show()

plot_class_distribution(train_gen)
plot_class_distribution(validation_gen)
plot_class_distribution(test_gen)
Model Training
1.	Load and modify the InceptionV3 model:
 	from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
out = base_model.output
pool = GlobalAveragePooling2D()(out)
pool = Dropout(0.5)(pool)
output = Dense(1024, activation='relu')(pool)
output = Dropout(0.5)(output)
predictions = Dense(train_gen.num_classes, activation='softmax')(output)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
2.	Train the model:
 	from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // validation_gen.batch_size,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)
Model Evaluation
1.	Evaluate the model on the test set:
 	test_loss, test_acc = model.evaluate(test_gen)
print(f'Test accuracy: {test_acc}')
2.	Generate confusion matrix:
 	from sklearn.metrics import confusion_matrix
import numpy as np

def get_predictions_and_labels(generator, model):
    all_preds = []
    all_labels = []
    for batch in generator:
        imgs, labels = batch
        preds = model.predict(imgs)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(labels, axis=1))
    return np.array(all_preds), np.array(all_labels)

y_pred, y_true = get_predictions_and_labels(test_gen, model)
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:\n', cm)
3.	Plot confusion matrix:
 	import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

class_names = list(train_gen.class_indices.keys())
plot_confusion_matrix(cm, class_names)
Additional Information Integration
This section demonstrates how to integrate an additional dataset that includes detailed information about the foods, such as name, origin, nutritional content, health benefits, and more.
1.	Load additional information:
 	import pandas as pd

info_df = pd.read_excel('path_to_additional_info.xlsx')
print(info_df.head())
2.	Merge predictions with additional information:
 	def get_food_info(prediction_index, info_df):
    food_name = class_names[prediction_index]
    food_info = info_df[info_df['Name'] == food_name]
    return food_info.to_dict(orient='records')[0] if not food_info.empty else {}

sample_image, sample_label = test_gen[0][0][0], test_gen[0][1][0]
sample_prediction = model.predict(sample_image[np.newaxis, ...])
predicted_class = np.argmax(sample_prediction)

food_info = get_food_info(predicted_class, info_df)
print(food_info)
Deployment
Deploy the model on a user-friendly and interactive platform using Streamlit.
1.	Install Streamlit:
 	pip install streamlit
2.	Create a Streamlit app:
 	import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_inception_model():
    model = load_model('best_model.keras')
    return model

@st.cache
def load_food_info():
    return pd.read_excel('path_to_additional_info.xlsx')

def get_food_info(prediction_index, info_df):
    food_name = class_names[prediction_index]
    food_info = info_df[info_df['Name'] == food_name]
    return food_info.to_dict(orient='records')[0] if not food_info.empty else {}

model = load_inception_model()
info_df = load_food_info()

st.title('Nigerian Foods and Snacks Classifier')
uploaded_file = st.file_uploader('Upload an image of the food', type=['jpg', 'png'])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(299, 299))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    food_info = get_food_info(predicted_class, info_df)

    st.write('### Predicted Class:', class_names[predicted_class])
    st.write('### Additional Information:')
    for key, value in food_info.items():
        st.write(f'**{key.capitalize()}**: {value}')
3.	Run the Streamlit app:
 	streamlit run app.py
Usage
1.	Fine-tune the best model:
 	best_model = model

callbacks_new = [
    ModelCheckpoint('best_newmodel.keras', save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
]

class_weights_new = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_new = dict(enumerate(class_weights_new))

history_new = best_model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=10,
    class_weight=class_weights_new,
    callbacks=callbacks_new
)

test_loss, test_acc = best_model.evaluate(test_gen)
print(f'Test accuracy: {test_acc}')
Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!
