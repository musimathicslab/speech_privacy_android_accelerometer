import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import seaborn as sns
import pickle
from scipy import signal
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


# Dataset used by CNN
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


def high_pass_filter(x, cutoff):
    """
      FILTRO PASSA ALTO
      it divide the signal of speech from that of movements
      :param x: one of the axis of the data
      :param cutoff: limit of the cutoff
      :return: the data with cutoff application
    """
    fs = len(x)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    f, t, zxx = signal.stft(x)

    for n in range(0, len(f)):
        if f[n] <= normal_cutoff:
            zxx[n] = 0

    a, rec = signal.istft(zxx)
    rec = list(rec)

    return rec


def get_spectrogram(waveform):
    """
       Function for generate a spectrogram from a signal
       :param waveform: the signal
       :return: the spectogram of signal
    """

    spectrogram = tf.signal.stft(
        waveform, frame_length=50, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def interpolate_1d_vector(vector, factor):
    """
        Function of interpolation
        Interpolate, i.e. upsample, a given 1D vector by a specific interpolation factor.
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


def set_element(file, dir):

    """
       Signal load and pre-elaboration
       :param file: name of the file to be processed
       :param dir : name of the file's directory
       :return: data vector that represent the signal
    """
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


def list_dir():
    """
      Cycle of signal elaboration
       :return: data vector that represent the signal
    """
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


def save_direct(direct, targ):
    """
      function that save a pre-elaborate dataset into a pickel
    :param direct: the pre-elaborate dataset to save
    :param targ: the target of the dataset to save
    """
    # Salvataggio DATASET & TARGET
    print(f'SALVATAGIO DATASET IN CORSO...')
    np.save('Dataset_Pickel/Dataset1800_86', direct)
    filehandler_out = open('Dataset_Pickel/Dataset1800_86.npy', 'wb')
    pickle.dump(direct, filehandler_out, protocol=4)
    filehandler_out.close()

    np.save('Dataset_Pickel/Target1800_86', targ)
    filehandler_out2 = open('Dataset_Pickel/Target1800_86.npy', 'wb')
    pickle.dump(targ, filehandler_out2, protocol=4)
    filehandler_out2.close()
    print(f'SALVATAGIO DATASET TERMINATO.')


def open_direct():
    """
    function that open a pre-saved dataset from the pickel
    :return: pre-elaborate dataset and target
    """
    # Recupero DATASET & TARGET
    print(f'RECUPERO DATASET IN CORSO...')
    deserialized_a = np.load('Dataset_Pickel/Dataset1800_86.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_a = open('Dataset_Pickel/Dataset1800_86.npy', 'rb')
    Direct2 = pickle.load(deserialized_a)
    deserialized_b = np.load('Dataset_Pickel/Target1800_86.npy', allow_pickle=True, mmap_mode='rb')
    deserialized_b = open('Dataset_Pickel/Target1800_86.npy', 'rb')
    Target2 = pickle.load(deserialized_b)
    print(f'RECUPERO DATASET TERMINATO.')

    return Direct2, Target2


if __name__ == '__main__':

    # CARICAMENTO DATASET
    # Dataset, Target = list_dir()
    # Salvataggio DATASET & TARGET
    # save_direct(Dataset, Target)
    # Recupero DATASET & TARGET
    Dataset, Target = open_direct()

    # CREAZIONE DI TRAIN-SET E TEST-SET
    x_train, x_test, y_train, y_test = train_test_split(Dataset, Target, test_size=0.2, stratify=Target)

    train_x = np.asarray(x_train)
    train_y = np.asarray(y_train)

    test_x = np.asarray(x_test)
    test_y = np.asarray(y_test)

    for spectrogram in Dataset:
        input_shape = spectrogram.shape

    # array mutation
    # Dataset = np.asarray(Dataset)

    # CNN Building
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_x)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(86)
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # Model Training
    EPOCHS = 20
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
    plt.figure(figsize=(30, 28))
    commands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 83, 84, 85, 86]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('CNN_Models/Result_Test53')
    plt.show()
