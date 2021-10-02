import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten




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

    x_train = tf.image.resize(x_train, [299, 299])
    x_test = tf.image.resize(x_test, [299, 299])

    train_x = np.asarray(x_train)
    train_y = np.asarray(y_train)

    test_x = np.asarray(x_test)
    test_y = np.asarray(y_test)



    # Estrazione spettogramma
    for spectrogram in Dataset:
        input_shape = spectrogram.shape
        #print('Input shape:', input_shape)

    Dataset = np.asarray(Dataset)

    # COSTRUZIONE DELLA CNN
    # create the base pre-trained model
    input_tensor = Input(shape=(299, 299, 3))

    base_model = InceptionV3(input_tensor=input_tensor,
                             input_shape=input_shape,
                             weights='imagenet',
                             include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    # let's add a fully-connected layer
    flat1 = Flatten()(x)
    dense1 = Dense(1024, activation='relu')(flat1)
    normal2 = BatchNormalization()(dense1)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(20, activation='softmax')(normal2)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()
    # compile the model (should be done *after* setting layers to non-trainable)
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
    commands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('ReteSpeechModels/Result_Test18')
    plt.show()
