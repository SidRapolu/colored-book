import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model, load_model


def images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


colored_images = images_from_folder('../dataset/colored')
gray_images = images_from_folder('../dataset/gray')

colored_images = np.array(colored_images, dtype=np.float32) / 255.0
gray_images = np.array(gray_images, dtype=np.float32) / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    gray_images, colored_images, test_size=0.2, random_state=42)


def create_unet():
    # shape of the images, 1 is grayscale
    inputs = Input((256, 256, 1))

    # Layer 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Layer 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Layer 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Layer 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Middle
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Layer 6
    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    # Layer 7
    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    # Layer 8
    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    # Layer 9
    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # The final layer uses a softmax activation function for multiclass segmentation
    # Here we assume the output is three channels for colored images (RGB)
    outputs = Conv2D(3, (1, 1), activation='softmax')(conv9)

    return Model(inputs=[inputs], outputs=[outputs])


model = create_unet()
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_test, y_test))

model.save('model.h5')


def predict_image(model, gray_image):
    gray_image = np.array(gray_image, dtype=np.float32) / 255.0
    # Need to add an extra dimension because the model expects batches of images
    gray_image = np.expand_dims(gray_image, axis=0)
    prediction = model.predict(gray_image)
    # Remove the extra batch dimension and scale up to 0-255 range
    colored_image = np.squeeze(prediction, axis=0) * 255.0
    return colored_image
