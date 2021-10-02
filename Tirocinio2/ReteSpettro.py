import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


directory = ['DatasetMobile1300/Dataset0', 'DatasetMobile1300/Dataset1',
             'DatasetMobile1300/Dataset2', 'DatasetMobile1300/Dataset3',
             'DatasetMobile1300/Dataset4', 'DatasetMobile1300/Dataset5',
             'DatasetMobile1300/Dataset6', 'DatasetMobile1300/Dataset7',
             'DatasetMobile1300/Dataset8', 'DatasetMobile1300/Dataset9']

#FILTRO PASSA ALTO
def highPassFilter(x,cutoff):
    fs=len(x)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    f,t,zxx = signal.stft(x)

    for n in range(0,len(f)):
        if f[n]<=normal_cutoff:
            zxx[n] =0

    a,rec = signal.istft(zxx)
    rec=list(rec)

    '''if (len(rec))>=fs:
        i=len(rec)-fs
        for n in range(0,i):
            rec.pop()'''

    return rec

#FUNZIONE PER OTTENERE SPETTROGRAMMA DA UN SEGNALE
def get_spectrogram(waveform):

    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

#FUNZIONE DI INTERPOLAZIONE
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

#CARICAMENTO DATASET E PRE-ELABORAZIONE
def setElement(file,dir):
    x=[]
    y=[]
    z=[]
    data=[]
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



    if len(x)>=256:

        # PRE-ELABORAZIONE
        x = list(interpolate_1d_vector(x, 3))
        y = list(interpolate_1d_vector(y, 3))
        z = list(interpolate_1d_vector(z, 3))

        x = highPassFilter(x,80)
        y = highPassFilter(y,80)
        z = highPassFilter(z,80)

        x = get_spectrogram(x)
        y = get_spectrogram(y)
        z = get_spectrogram(z)

        elem = []

        for n in range(0, len(x)):
            for i in range(0, len(x[n])):
                elem.append([ math.sqrt(x[n][i]), math.sqrt(y[n][i]), math.sqrt(z[n][i]) ])
            data.append(elem)
            elem = []
        data = np.asarray(data)
        pil_img = tf.keras.preprocessing.image.array_to_img(data)
        data = tf.keras.preprocessing.image.img_to_array(pil_img)
        data = tf.image.resize(data, [224, 224])
        print(f'{dir} {file} : TERMINATO')
    return data


def listDir():
    data = []
    target = []
    cont = -1
    for dir in directory:
        cont = cont + 1
        folder = os.listdir(dir)
        for file in folder:
            x = setElement(file,dir)
            if len(x)>0:
                data.append(x)
                target.append(cont)
        print(f'{dir} TERMINATO')


    return data,target





if __name__ == '__main__':

    # CARICAMENTO DATASET
    print('CARICAMENTO DATASET IN CORSO... ')
    Dataset, Target = listDir()
    print('CARICAMENTO DATASET TERMINATO ')
    # CREAZIONE DI TRAIN-SET E TEST-SET
    x_train, x_test, y_train, y_test = train_test_split(Dataset, Target, test_size=0.2, stratify=Target)

    train_x = np.asarray(x_train)
    train_y = np.asarray(y_train)

    test_x = np.asarray(x_test)
    test_y = np.asarray(y_test)

    for spectrogram in Dataset:
        input_shape = spectrogram.shape
        #print('Input shape:', input_shape)

    Dataset =np.asarray(Dataset)

    # COSTRUZIONE DELLA CNN
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
        layers.Dense(10)
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # ADDESTRAMENTO DEL MODELLO
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
    plt.figure(figsize=(10, 8))
    commands = [0,1,2,3,4,5,6,7,8,9]
    commands = np.asarray(commands)
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    model.save('ReteMobile1300')
    plt.show()


