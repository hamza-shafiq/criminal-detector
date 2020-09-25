import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json


model_path = "ML/models/keras"
training_data_path = "ML/data"
test_data_path = ""
image_path = "ML/test/1.jpg"


def train_model(training_data_path=training_data_path, model_path=model_path):
    train_datagen = ImageDataGenerator(
        shear_range=10,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        training_data_path + '/train',
        batch_size=32,
        class_mode='binary',
        target_size=(224, 224))

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    validation_generator = validation_datagen.flow_from_directory(
        training_data_path + '/val',
        shuffle=False,
        class_mode='binary',
        target_size=(224, 224))

    conv_base = ResNet50(
        include_top=False,
        weights='imagenet')

    for layer in conv_base.layers:
        layer.trainable = False

    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(2, activation='softmax')(x)
    model = Model(conv_base.input, predictions)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  epochs=1,
                                  validation_data=validation_generator)

    # architecture and weights to HDF5
    model.save(model_path + '/model.h5')

    # architecture to JSON, weights to HDF5
    model.save_weights(model_path + '/weights.h5')
    with open(model_path + '/architecture.json', 'w') as f:
        f.write(model.to_json())


def predict_criminal(image_path):

    # architecture and weights from HDF5
    model = load_model(model_path + '/model.h5')

    # architecture from JSON, weights from HDF5
    with open(model_path + '/architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(model_path + '/weights.h5')

    validation_img_paths = [image_path]
    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    validation_batch = np.stack([preprocess_input(np.array(img.resize((224, 224))))
                                 for img in img_list])

    pred_probs = model.predict(validation_batch)
    return {"Citizen": round(pred_probs[0, 0] * 100, 2),
            "Criminal": round(pred_probs[0, 1] * 100, 2)
            }


# print(predict_criminal(model_path=model_path, image_path=image_path))
