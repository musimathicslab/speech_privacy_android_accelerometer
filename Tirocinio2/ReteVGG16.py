import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, MaxPooling2D
from keras.layers import Flatten
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
    model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in model.layers:
        layer.trainable = False

    normal1 = BatchNormalization()(model.layers[-2].output)
    pool1 = MaxPooling2D()(normal1)
    flat1 = Flatten()(pool1)
    class1 = Dense(1024, activation='relu')(flat1)
    normal2 = BatchNormalization()(class1)
    class2 = Dense(2048, activation='relu')(normal2)
    normal3 = BatchNormalization()(class2)
    output = Dense(50, activation='softmax')(normal3)

    model = Model(inputs=model.inputs, outputs=output)

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],)

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
    commands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('ReteSpeechModels/Result_Test26')
    plt.show()
