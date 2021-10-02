import csv
import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Input
import math
import os
import scipy.interpolate
from scipy import signal

def open_direct():
    # Recupero DATASET & TARGET
    print(f'RECUPERO DATASET IN CORSO...')
    deserialized_a = np.load('Dataset_Pickel/Dataset1800_60.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_a = open('Dataset_Pickel/Dataset1800_60.npy', 'rb')
    Direct2 = pickle.load(deserialized_a)
    deserialized_b = np.load('Dataset_Pickel/Target1800_60.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_b = open('Dataset_Pickel/Target1800_60.npy', 'rb')
    Target2 = pickle.load(deserialized_b)
    print(f'RECUPERO DATASET TERMINATO.')

    return Direct2, Target2


directory = ['DatasetSpeechUnit/Dataset_A', 'DatasetSpeechUnit/Dataset_All', 'DatasetSpeechUnit/Dataset_As',
             'DatasetSpeechUnit/Dataset_Be',  'DatasetSpeechUnit/Dataset_Beau',  'DatasetSpeechUnit/Dataset_Best',
             'DatasetSpeechUnit/Dataset_Birds', 'DatasetSpeechUnit/Dataset_Bite', 'DatasetSpeechUnit/Dataset_Book',
             'DatasetSpeechUnit/Dataset_But',
             'DatasetSpeechUnit/Dataset_By',  'DatasetSpeechUnit/Dataset_Can', 'DatasetSpeechUnit/Dataset_Cant',
             'DatasetSpeechUnit/Dataset_Co',  'DatasetSpeechUnit/Dataset_Cy',
             'DatasetSpeechUnit/Dataset_Day',  'DatasetSpeechUnit/Dataset_Der', 'DatasetSpeechUnit/Dataset_Do',
             'DatasetSpeechUnit/Dataset_Done', 'DatasetSpeechUnit/Dataset_Dont',  'DatasetSpeechUnit/Dataset_Drink',
             'DatasetSpeechUnit/Dataset_Eye',  'DatasetSpeechUnit/Dataset_Fea', 'DatasetSpeechUnit/Dataset_Feeds',
             'DatasetSpeechUnit/Dataset_Fire', 'DatasetSpeechUnit/Dataset_Flock', 'DatasetSpeechUnit/Dataset_Free',
             'DatasetSpeechUnit/Dataset_Get', 'DatasetSpeechUnit/Dataset_Gether',
             'DatasetSpeechUnit/Dataset_Hand', 'DatasetSpeechUnit/Dataset_Have','DatasetSpeechUnit/Dataset_Him',
             'DatasetSpeechUnit/Dataset_Hol', 'DatasetSpeechUnit/Dataset_Home',
             'DatasetSpeechUnit/Dataset_Horse', 'DatasetSpeechUnit/Dataset_I', 'DatasetSpeechUnit/Dataset_If',
             'DatasetSpeechUnit/Dataset_In', 'DatasetSpeechUnit/Dataset_Is',
             'DatasetSpeechUnit/Dataset_It', 'DatasetSpeechUnit/Dataset_Its', 'DatasetSpeechUnit/Dataset_Judge',
             'DatasetSpeechUnit/Dataset_Lead', 'DatasetSpeechUnit/Dataset_Li',
             'DatasetSpeechUnit/Dataset_Like', 'DatasetSpeechUnit/Dataset_Lunch', 'DatasetSpeechUnit/Dataset_Ma',
             'DatasetSpeechUnit/Dataset_Make', 'DatasetSpeechUnit/Dataset_Mo', 'DatasetSpeechUnit/Dataset_Ne',
             'DatasetSpeechUnit/Dataset_No', 'DatasetSpeechUnit/Dataset_Ny', 'DatasetSpeechUnit/Dataset_O',
             'DatasetSpeechUnit/Dataset_Of', 'DatasetSpeechUnit/Dataset_Off', 'DatasetSpeechUnit/Dataset_Place',
             'DatasetSpeechUnit/Dataset_Po', 'DatasetSpeechUnit/Dataset_Pre', 'DatasetSpeechUnit/Dataset_Put',
             'DatasetSpeechUnit/Dataset_Right', 'DatasetSpeechUnit/Dataset_Rons', 'DatasetSpeechUnit/Dataset_Rrow',
             'DatasetSpeechUnit/Dataset_Self', 'DatasetSpeechUnit/Dataset_Sent', 'DatasetSpeechUnit/Dataset_Some',
             'DatasetSpeechUnit/Dataset_Sty', 'DatasetSpeechUnit/Dataset_Such', 'DatasetSpeechUnit/Dataset_Ter',
             'DatasetSpeechUnit/Dataset_That', 'DatasetSpeechUnit/Dataset_The', 'DatasetSpeechUnit/Dataset_Ther',
             'DatasetSpeechUnit/Dataset_There', 'DatasetSpeechUnit/Dataset_Thing', 'DatasetSpeechUnit/Dataset_Til',
             'DatasetSpeechUnit/Dataset_Time', 'DatasetSpeechUnit/Dataset_To', 'DatasetSpeechUnit/Dataset_Too',
             'DatasetSpeechUnit/Dataset_Ty', 'DatasetSpeechUnit/Dataset_Un', 'DatasetSpeechUnit/Dataset_Ver',
             'DatasetSpeechUnit/Dataset_Wa', 'DatasetSpeechUnit/Dataset_Want', 'DatasetSpeechUnit/Dataset_Ways',
             'DatasetSpeechUnit/Dataset_What', 'DatasetSpeechUnit/Dataset_You', 'DatasetSpeechUnit/Dataset_Your']


# FILTRO PASSA ALTO
# it divide the signal of speech from that of movements
def high_pass_filter(x, cutoff):
    fs = len(x)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    f, t, zxx = signal.stft(x)

    for n in range(0, len(f)):
        if f[n] <= normal_cutoff:
            zxx[n] = 0

    a, rec = signal.istft(zxx)
    rec = list(rec)

    '''if (len(rec))>=fs:
        i=len(rec)-fs
        for n in range(0,i):
            rec.pop()'''

    return rec


# Function for generate a spectrogram from a signal
def get_spectrogram(waveform):

    spectrogram = tf.signal.stft(
        waveform, frame_length=50, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


# Function of interpolation
def interpolate_1d_vector(vector, factor):
    """
    Interpolate, i.e. upsample,
     a given 1D vector by a specific interpolation factor.
    :param vector: 1D data vector
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    x = np.arange(np.size(vector))
    y = vector
    f = scipy.interpolate.interp1d(x, y)

    x_extended_by_factor = np.linspace(x[0], x[-1], np.size(x) * factor)
    y_interpolated = np.zeros(np.size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_interpolated[i] = f(x)
        i += 1

    return y_interpolated


# Signal load and pre-elaboration
def set_element(file, dir):
    x = []
    y = []
    z = []
    data = []
    with open(dir + '/' + file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            if 'X' not in line:
                elem = float(line[1])
                x.append(elem)
                elem = float(line[2])
                y.append(elem)
                elem = float(line[3])
                z.append(elem)

    if len(x)>=50:

        # PRE-ELABORAZIONE
        x = list(interpolate_1d_vector(x, 3))
        y = list(interpolate_1d_vector(y, 3))
        z = list(interpolate_1d_vector(z, 3))

        x = high_pass_filter(x, 80)
        y = high_pass_filter(y, 80)
        z = high_pass_filter(z, 80)

        x = get_spectrogram(x)
        y = get_spectrogram(y)
        z = get_spectrogram(z)

        elem = []

        for n in range(0, len(x)):
            for i in range(0, len(x[n])):
                elem.append([math.sqrt(x[n][i]), math.sqrt(y[n][i]), math.sqrt(z[n][i])])
            data.append(elem)
            elem = []
        data = np.asarray(data)
        pil_img = tf.keras.preprocessing.image.array_to_img(data)
        data = tf.keras.preprocessing.image.img_to_array(pil_img)
        data = tf.image.resize(data, [224, 224])
        print(f'{dir} {file} : TERMINATO')
    return data


# Cycle of signal elaboration
def list_dir():
    data = []
    target = []
    cont = -1
    print('CARICAMENTO DATASET IN CORSO... ')

    for dir in directory:
        cont = cont + 1
        folder = os.listdir(dir)
        for file in folder:
            x = set_element(file, dir)
            if len(x) > 0:
                data.append(x)
                target.append(cont)
        print(f'{dir} TERMINATO')
    print('CARICAMENTO DATASET TERMINATO ')

    return data, target


if __name__ == '__main__':

    # DATASET & TARGET OPEN
    # Dataset, Target = open_direct()
    # DATASET LOAD
    Dataset, Target = list_dir()

    # Creation OF TRAIN-SET AND TEST-SET
    x_train, x_test, y_train, y_test = train_test_split(Dataset, Target, test_size=0.2, stratify=Target)

    train_x = np.asarray(x_train)
    train_y = np.asarray(y_train)

    test_x = np.asarray(x_test)
    test_y = np.asarray(y_test)

    for spectrogram in Dataset:
        input_shape = spectrogram.shape

    Dataset = np.asarray(Dataset)

    # CNN BUILDING
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_x)
    input_tensor = Input(shape=(224, 224, 3))
    model = models.Sequential([
        layers.Input(shape=input_shape, tensor=input_tensor),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu', name='conv1'),
        layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2'),
        layers.Conv2D(64, 3, padding='same', activation='relu', name='conv3'),
        layers.MaxPooling2D(name='pool1'),
        layers.Conv2D(128, 3, padding='same', activation='relu', name='conv4'),
        layers.Conv2D(256, 3, padding='same', activation='relu', name='conv5'),
        layers.MaxPooling2D(name='pool2'),
        layers.Flatten(),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout1'),
        layers.Dense(512, activation='relu', name='dense2'),
        layers.Dense(1024, activation='relu', name='dense3'),
        layers.Dropout(0.2, name='dropout2'),
        layers.Dense(86, name='dense4')
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # MODEL TRAINING
    EPOCHS = 50
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
    plt.figure(figsize=(16, 14))
    commands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 83, 84, 85, 86]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('ReteSpeechModels/Result_Test36')
    plt.show()
