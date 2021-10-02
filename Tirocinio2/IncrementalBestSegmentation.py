import csv
import math
import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf


# FILTRO PASSA ALTO
def highPassFilter(x, cutoff):
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


# FUNZIONE PER OTTENERE SPETTROGRAMMA DA UN SEGNALE
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=50, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


# FUNZIONE DI INTERPOLAZIONE
def interpolate_1d_vector(vector, factor):
    """
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

# CARICAMENTO Segnale E PRE-ELABORAZIONE
def setElement(x, y, z):
    x = x
    y = y
    z = z
    data = []

    if len(x) >= 50:

        x = list(interpolate_1d_vector(x, 3))
        y = list(interpolate_1d_vector(y, 3))
        z = list(interpolate_1d_vector(z, 3))

        x = highPassFilter(x, 80)
        y = highPassFilter(y, 80)
        z = highPassFilter(z, 80)

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
        print(f'Load file : END')
    return data


def listDir(x, y, z):
    data = []
    w = setElement(x, y, z)
    if len(w) > 0:
        data.append(w)
    data = np.asarray(data)
    return data


def percPrediction(pred):
    e_x = np.exp(pred - np.max(pred))
    return e_x / e_x.sum()


def searchSpeech(x, y, z):

    print(f'x segment: {len(x)}')
    print(f'y segment: {len(y)}')
    print(f'z segment: {len(z)}')
    speech = listDir(x, y, z)
    print(f'Search Speech')
    loaded_model = tf.keras.models.load_model('ReteSpeechModels/Result_Test26')
    prediction = loaded_model.predict(speech)
    y_pred = np.argmax(prediction)
    prediction_perc = percPrediction(prediction)
    acc_y_pred = prediction_perc[0][y_pred]*100
    prediction_media = 0.0
    for e in prediction_perc[0]:
        if e != prediction_perc[0][y_pred]:
            prediction_media = prediction_media + e

    prediction_media = prediction_media/(len(prediction_perc[0])-1)
    efficacy_pred = acc_y_pred-(prediction_media*100)
    print(f'speech: {y_pred}, -accuracy: {int(acc_y_pred)}%, -efficacy: {efficacy_pred}%')
    print(f']')

    return y_pred, acc_y_pred, efficacy_pred


def segmentationDir(t, x, y, z, i, temps):
    x1 = []
    y1 = []
    z1 = []
    stop = False
    temp = float(temps + 1.7)
    i = i

    while i < len(x) and t[i] < temp:
        x1.append(x[i])
        y1.append(y[i])
        z1.append(z[i])
        i = i+1

    if len(x1) >= 50:
        print(f'Segmentation: 0[')
        y_pred, acc_y_pred, efficacy_pred = searchSpeech(x1, y1, z1)
        print(f']')

        u = 1
        while stop != True and i < len(x):
            print(f'Segmentation: {u}[')
            u = u + 1
            temp = temp + 0.1
            while i < len(x) and t[i] < temp:
                x1.append(x[i])
                y1.append(y[i])
                z1.append(z[i])
                i = i + 1

            y1_pred, acc1_y_pred, efficacy1_pred = searchSpeech(x1, y1, z1)

            if efficacy1_pred < efficacy_pred:
                stop = True
                temp = temp - 0.1

            if efficacy1_pred == efficacy_pred:
                if acc1_y_pred >= acc_y_pred:
                    y_pred = y1_pred
                    acc_y_pred = acc1_y_pred
                    efficacy_pred = efficacy1_pred
                if acc1_y_pred < acc_y_pred:
                    stop = True
                    temp = temp - 0.1

            if efficacy1_pred > efficacy_pred:
                y_pred = y1_pred
                acc_y_pred = acc1_y_pred
                efficacy_pred = efficacy1_pred

    if len(x1) < 50:
        i = len(x)
        y_pred = -1
        acc_y_pred = 0
        efficacy_pred = 0
        temp = 0
    return i, y_pred, acc_y_pred, efficacy_pred, temp


def openDir(directory, file):
    t = []
    x = []
    y = []
    z = []
    data = []
    acc = []
    efficacy = []

    with open(directory + '/' + file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            if 'TIME' not in line:
                elem = float(line[0])
                t.append(elem)
                elem = float(line[1])
                x.append(elem)
                elem = float(line[2])
                y.append(elem)
                elem = float(line[3])
                z.append(elem)

    i = 0
    tmp = 0.0
    u = 1
    while i < len(x):
        print(f'Iteration: {u}[')
        u = u+1
        i1, y_pred, acc_y_pred, efficacy_pred, t1 = segmentationDir(t, x, y, z, i, tmp)
        i = i1
        if y_pred != -1 and acc_y_pred != 0:
            print(f'i: {i1}')
            print(f'time: {t1}')
            print(f' -speech: {y_pred}, -accuracy: {acc_y_pred}, -efficacy: {efficacy_pred}%')
            print(f']')
            tmp = t1
            data.append(y_pred)
            acc.append(acc_y_pred)
            efficacy.append(efficacy_pred)

    media_acc = 0
    media_efficacy = 0
    for e in acc:
        media_acc = media_acc + e
    for o in efficacy:
        media_efficacy = media_efficacy + o

    media_acc = media_acc / len(acc)
    media_efficacy = media_efficacy / len(efficacy)

    return data, media_acc, media_efficacy


if __name__ == '__main__':

    print("\nOpen file")

    # ELABORAZIONE DEL FILE
    speeches, accuracy, efficacy = openDir('SegmentationFile', 'frase1.csv')

    # INVIO RISULTATI
    print(f'SPEECH Found :')
    for element in speeches:
        print(f'{element}')

    print(f'Accuracy: {accuracy}%')
    print(f'Efficacy: {efficacy}%')
