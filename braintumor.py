{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1PykYvxQntgZk5YAk2IsdErKOpbgQU6Tn",
      "authorship_tag": "ABX9TyNFbuWI3Ad4VVkbuBl0PZ/d",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/challavishnu123/DATA-ANALYSIS/blob/main/braintumor.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "Z_FMKtQL1uCv"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "# Define image size and batch size\n",
        "IMG_SIZE = 224\n",
        "BATCH_SIZE = 32"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define data generators for train, validation and test sets\n",
        "train_datagen = ImageDataGenerator(\n",
        "    rescale=1./255,\n",
        "    validation_split=0.2\n",
        ")\n",
        "\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "    '/content/drive/MyDrive/Brain_Tumor_Detection/train',\n",
        "    target_size=(IMG_SIZE, IMG_SIZE),\n",
        "    batch_size=BATCH_SIZE,\n",
        "    class_mode='binary',\n",
        "    subset='training'\n",
        ")\n",
        "\n",
        "val_generator = train_datagen.flow_from_directory(\n",
        "    '/content/drive/MyDrive/Brain_Tumor_Detection/train',\n",
        "    target_size=(IMG_SIZE, IMG_SIZE),\n",
        "    batch_size=BATCH_SIZE,\n",
        "    class_mode='binary',\n",
        "    subset='validation'\n",
        ")\n",
        "\n",
        "test_datagen = ImageDataGenerator(rescale=1./255)\n",
        "\n",
        "test_generator = test_datagen.flow_from_directory(\n",
        "    '/content/drive/MyDrive/Brain_Tumor_Detection/test',\n",
        "    target_size=(IMG_SIZE, IMG_SIZE),\n",
        "    batch_size=BATCH_SIZE,\n",
        "    class_mode='binary'\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tqYkKfoI20S4",
        "outputId": "0c599a5f-70e0-4cf9-a63e-c1f643f1d879"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 2400 images belonging to 2 classes.\n",
            "Found 600 images belonging to 2 classes.\n",
            "Found 60 images belonging to 1 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the model\n",
        "model = keras.Sequential([\n",
        "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),\n",
        "    layers.MaxPooling2D((2, 2)),\n",
        "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
        "    layers.MaxPooling2D((2, 2)),\n",
        "    layers.Conv2D(128, (3, 3), activation='relu'),\n",
        "    layers.MaxPooling2D((2, 2)),\n",
        "    layers.Flatten(),\n",
        "    layers.Dense(128, activation='relu'),\n",
        "    layers.Dense(1, activation='sigmoid')\n",
        "])"
      ],
      "metadata": {
        "id": "c9WicXig9dgc"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#compile the model\n",
        "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "a0GtCf1kAwvF"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history=model.fit(train_generator,validation_data=val_generator,epochs=5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BNZ9PcHxCLt2",
        "outputId": "8ec65bba-2931-43e3-e25c-5206494348de"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "75/75 [==============================] - 335s 4s/step - loss: 0.2957 - accuracy: 0.8821 - val_loss: 0.1924 - val_accuracy: 0.9233\n",
            "Epoch 2/5\n",
            "75/75 [==============================] - 262s 3s/step - loss: 0.1618 - accuracy: 0.9442 - val_loss: 0.1010 - val_accuracy: 0.9633\n",
            "Epoch 3/5\n",
            "75/75 [==============================] - 259s 3s/step - loss: 0.0792 - accuracy: 0.9758 - val_loss: 0.0697 - val_accuracy: 0.9733\n",
            "Epoch 4/5\n",
            "75/75 [==============================] - 252s 3s/step - loss: 0.0451 - accuracy: 0.9875 - val_loss: 0.0576 - val_accuracy: 0.9817\n",
            "Epoch 5/5\n",
            "75/75 [==============================] - 254s 3s/step - loss: 0.0315 - accuracy: 0.9904 - val_loss: 0.0182 - val_accuracy: 0.9950\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save(\"model.h5\",\"label.txt\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S4UNWR2fFs5-",
        "outputId": "a0f134a5-d7fb-4f96-f646-c48601a8a062"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import load_model\n",
        "from tensorflow.keras.preprocessing import image\n",
        "import numpy as np\n",
        "# Load and saved model\n",
        "model = load_model('/content/model.h5')\n",
        "test_image_path = '/content/drive/MyDrive/Brain_Tumor_Detection/train/no/No15.jpg'\n",
        "img = image.load_img(test_image_path, target_size=(224,224))\n",
        "img_array = image.img_to_array(img)\n",
        "img_array = np.expand_dims(img_array, axis=0)\n",
        "img_array /= 255.\n",
        "predication = model.predict(img_array)\n",
        "if predication<0.5:\n",
        "  print(\"prediction: No tumor (Probability\", predication[0][0],\")\")\n",
        "else:\n",
        "  print(\"prediction:  tumor (Probability\", predication[0][0],\")\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AVRnIyzrGFBx",
        "outputId": "35aa7d7e-221c-417f-e1d3-8fcdfbd379b2"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 103ms/step\n",
            "prediction: No tumor (Probability 2.7669703e-05 )\n"
          ]
        }
      ]
    }
  ]
}