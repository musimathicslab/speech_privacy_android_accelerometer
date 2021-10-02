import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras.layers import Input
from keras.layers.normalization import BatchNormalization


def openDirect():
    # Recupero DATASET & TARGET
    print(f'RECUPERO DATASET IN CORSO...')
    deserialized_a = np.load('Dataset_Pickel/Dataset1800_60.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_a = open('Dataset_Pickel/Dataset1800_60.npy', 'rb')
    direct2 = pickle.load(deserialized_a)
    deserialized_b = np.load('Dataset_Pickel/Target1800_60.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_b = open('Dataset_Pickel/Target1800_60.npy', 'rb')
    target2 = pickle.load(deserialized_b)
    print(f'RECUPERO DATASET TERMINATO.')

    return direct2, target2


if __name__ == '__main__':

    # Recupero DATASET & TARGET
    Dataset, Target = openDirect()

    # CREAZIONE DI TRAIN-SET E TEST-SET
    x_train, x_test, y_train, y_test = train_test_split(Dataset, Target, test_size=0.2, stratify=Target)

    train_x = np.asarray(x_train)
    train_y = np.asarray(y_train)

    test_x = np.asarray(x_test)
    test_y = np.asarray(y_test)

    # Estrazione spettogramma
    for spectrogram in Dataset:
        input_shape = spectrogram.shape
        # print('Input shape:', input_shape)

    Dataset = np.asarray(Dataset)

    # COSTRUZIONE DELLA CNN
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(include_top=False,
                          weights="imagenet",
                          input_tensor=input_tensor,
                          input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    pool1 = MaxPooling2D()(x)
    flat1 = Flatten()(pool1)
    # let's add a fully-connected layer
    dense1 = Dense(1024, activation='relu')(flat1)
    normal1 = BatchNormalization()(dense1)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(20, activation='softmax')(normal1)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'], )

    # ADDESTRAMENTO DEL MODELLO
    EPOCHS = 40
    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=EPOCHS
    )
    y_pred = np.argmax(model.predict(test_x), axis=1)
    y_true = test_y
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    commands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('ReteSpeechModels/Result_Test14')
    plt.show()
