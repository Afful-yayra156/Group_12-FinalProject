#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os
import json
from pathlib import Path
import zipfile
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[122]:


pip install --upgrade tensorflow


# In[123]:


# Install and configure Kaggle API
get_ipython().system('pip install kaggle')


# In[124]:


kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)


# In[125]:


kaggle_json_path = kaggle_dir / "kaggle.json"


# In[126]:


with open("C:/Users/user/OneDrive - Ashesi University/intro to ai/kaggle.json") as f:
    kaggle_creds = json.load(f)

with open(kaggle_json_path, 'w') as f:
    json.dump(kaggle_creds, f)

kaggle_json_path.chmod(0o600)


# In[127]:


# Download and extract dataset
get_ipython().system('kaggle datasets download -d peaceedogun/nigerian-foods-and-snacks-multiclass')


# In[128]:


zip_file_path = Path("C:/Users/user/OneDrive - Ashesi University/intro to ai/nigerian-foods-and-snacks-multiclass.zip")
extract_dir = Path("C:/Users/user/OneDrive - Ashesi University/intro to ai/nigerian-foods-and-snacks")
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)


# In[129]:


train_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\train"
test_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\test"
val_dir = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass\validation"
whole_data = r"C:\Users\user\OneDrive - Ashesi University\intro to ai\nigerian-foods-and-snacks\naija_foods\content\naija_foods_multiclass"


# In[130]:


# Function to check and remove corrupt images
from PIL import Image

def check_and_remove_corrupt_images(directory):
    corrupt_images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError, OSError) as e:
                print(f'Removing bad file: {filepath}')
                corrupt_images.append(filepath)
                os.remove(filepath)  # Remove the corrupted file
    return corrupt_images

corrupt_images = check_and_remove_corrupt_images(whole_data)
print(f'Found and removed {len(corrupt_images)} corrupt images.')


# In[131]:


# Loading and preprocessing data
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


# In[132]:


# Function to plot class distribution
def plot_class_distribution(generator, title):
    class_counts = np.bincount(generator.classes)
    class_names = list(generator.class_indices.keys())
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names, y=class_counts)
    plt.xticks(rotation=90)
    plt.xlabel("Class Names")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_class_distribution(train_gen, "Class Distribution for Training Data")
plot_class_distribution(validation_gen, "Class Distribution for Validation Data")
plot_class_distribution(test_gen, "Class Distribution for Testing Data")


# In[133]:


# Model setup
def create_model(learning_rate=0.001, dropout_rate=0.5, num_classes=None):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[134]:


# Create and compile the model
model = create_model(num_classes=train_gen.num_classes)


# In[135]:


# Callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
]

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))


# In[136]:


# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // validation_gen.batch_size,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)


# In[139]:


# import tensorflow as tf
# model_save_path = "C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\best_model.keras"
# model.save(model_save_path)
# print(f"Model saved to {model_save_path}")


# In[140]:


# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[141]:


# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[142]:


import numpy as np
from sklearn.base import BaseEstimator
from keras.callbacks import EarlyStopping

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import make_scorer
import streamlit as st
from PIL import Image



# Define the model creation function
def create_model(learning_rate=0.001, dropout_rate=0.5, num_classes=10):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model





# In[ ]:





# In[143]:


# Wrapper class for Keras model
class KerasClassifierWrapper(BaseEstimator):
    def __init__(self, learning_rate=0.001, dropout_rate=0.5, batch_size=16, epochs=10):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
    
    def fit(self, X, y):
        self.model = create_model(learning_rate=self.learning_rate, dropout_rate=self.dropout_rate, num_classes=len(np.unique(y)))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2, callbacks=[early_stopping])
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)
    
    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]  # Return accuracy

# Define custom scorer
def evaluate_model(estimator, X, y):
    return estimator.score(X, y)

custom_scorer = make_scorer(evaluate_model, greater_is_better=True)


# In[ ]:





# In[ ]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.0001],
    'dropout_rate': [0.5],
    'batch_size': [16],
    'epochs': [5]
}

# Wrap the Keras model in a Scikit-learn compatible interface
keras_model = KerasClassifierWrapper()

# Create the GridSearchCV object
grid = GridSearchCV(
    estimator=keras_model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=3,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1  # Parallelize the search
)

# Assume train_gen and validation_gen are your training and validation data generators
# You need to preprocess the data from your generators

def preprocess_generator(generator):
    X, y = [], []
    for batch in generator:
        if len(batch[0]) != generator.batch_size:
            continue  # Skip batches that do not match the generator's batch size
        X.append(batch[0])
        y.append(batch[1])
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

# Example usage
X_train, y_train = preprocess_generator(train_gen)
X_val, y_val = preprocess_generator(validation_gen)

# Fit GridSearchCV
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")

# To get the best model
best_model = grid_result.best_estimator_


# In[ ]:





# In[144]:


from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd


# In[145]:


# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\user\\best_model.keras")


# In[146]:


# Load additional information from Excel
info_df = pd.read_excel("C:/Users/user/OneDrive - Ashesi University/intro to ai/Nigerianfood_additionalinfo.xlsx")


# In[147]:


def preprocess_image(img):
    """
    Preprocess the image for prediction.
    """
    img = img.resize((299, 299))  # Resize image to the model's expected input size
    img_array = np.array(img, dtype=np.float32)  # Convert image to numpy array with float32 type
    if img_array.ndim == 2:  # Check if image is grayscale
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert grayscale to RGB
    img_array /= 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# In[148]:


def get_food_name(predicted_class):
    """
    Retrieve the food name based on the predicted class index.
    """
    food_names = [
        'Abacha and Ugba', 'Akara and Eko', 'Amala and Gbegiri-Ewedu', 'Asaro', 'Boli(Bole)', 
        'Chin Chin', 'Egusi Soup', 'Ewa-Agoyin', 'Fried plantains(Dodo)', 'Jollof Rice', 
        'Meat Pie', 'Moin-moin', 'Nkwobi', 'Okro Soup', 'Pepper Soup', 'Puff Puff', 
        'Suya', 'Vegetable Soup'
    ]
    return food_names[predicted_class]


# In[149]:


def get_additional_info(food_name):
    """
    Retrieve additional information from the DataFrame based on the food name.
    """
    if food_name in info_df['food_name'].values:
        info = info_df[info_df['food_name'] == food_name].iloc[0]
        return {
            'Origin or State': info['Origin_or_State'],
            'Popular Countries': info['Pop_Countries'],
            'Health Benefits': info['Health_Benefits'],
            'Calories': info['calories'],
            'Nutrient Ratio': info['Nutrient_Ratio'],
            'Ingredients': info['Ingredients'],
            'Protein Content': info['Protein_Content'],
            'Fat Content': info['Fat_Content'],
            'Carbohydrate Content': info['Carbohydrate_Content'],
            'Allergens': info['Allergens'],
            'Mineral Content': info['Mineral-Content'],
            'Vitamin Content': info['Vitamin_Content'],
            'Suitability': info['Suitability'],
            'Fiber Content': info['Fiber_Content']
        }
    return None


# In[150]:


def predict_and_get_info(image):
    """
    Predict the food and retrieve additional information.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Retrieve additional information
    food_name = get_food_name(predicted_class)
    additional_info = get_additional_info(food_name)

    return food_name, additional_info


# In[155]:


image_path = "C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\Nigeria\\Nkwobi\\20180617_095955.jpg"


# In[156]:


# Load the image
image = Image.open(image_path)


# In[157]:


# Predict and get additional information
food_name, additional_info = predict_and_get_info(image)


# In[ ]:





# In[158]:


# Display results
print(f"Predicted Food: {food_name}")
if additional_info:
    for key, value in additional_info.items():
        print(f"{key}: {value}")
else:
    print("No additional information available.")


# In[ ]:





# In[ ]:





# In[ ]:




