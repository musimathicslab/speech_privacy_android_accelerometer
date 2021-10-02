import csv
import math
import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf


#FILTRO PASSA ALTO
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


#FUNZIONE PER OTTENERE SPETTROGRAMMA DA UN SEGNALE
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=50, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


#FUNZIONE DI INTERPOLAZIONE
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

#CARICAMENTO Segnale E PRE-ELABORAZIONE
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

    prediction_media = prediction_media / (len(prediction_perc[0]) - 1)
    efficacy_pred = acc_y_pred - (prediction_media * 100)
    print(f'speech: {y_pred}, -accuracy: {int(acc_y_pred)}%, -efficacy:{efficacy_pred}')
    print(f']')

    return y_pred, acc_y_pred, efficacy_pred


def segmentationDir(t, x, y, z, i, temps):
    x1 = []
    y1 = []
    z1 = []
    seg_pred = []
    acc_seg = []
    efficacy_seg = []
    temp = float(temps)
    i = i

    while i < len(x) and t[i] < temp:
        x1.append(x[i])
        y1.append(y[i])
        z1.append(z[i])
        i = i+1

    if len(x1) >= 50:
        print(f'Segmentation: 0[')
        y_pred, acc_y_pred, efficacy_pred = searchSpeech(x1, y1, z1)
        seg_pred.append(y_pred)
        acc_seg.append(acc_y_pred)
        efficacy_seg.append(efficacy_pred)
        print(f']')

        u = 1
        while i < len(x):
            print(f'Segmentation: {u}[')
            u = u + 1
            temp = temp + temps
            while i < len(x) and t[i] < temp:
                x1.append(x[i])
                y1.append(y[i])
                z1.append(z[i])
                i = i + 1

            y1_pred, acc1_y_pred, efficacy1_pred = searchSpeech(x1, y1, z1)
            seg_pred.append(y1_pred)
            acc_seg.append(acc1_y_pred)
            efficacy_seg.append(efficacy1_pred)

    if len(x1) < 50:
        seg_pred = -1
        acc_seg = 0
        efficacy_seg = 0

    return seg_pred, acc_seg, efficacy_seg


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

    f = len(t)-1
    temp = t[f]/2
    nt = 3
    u = 1
    while temp > 0.5:
        print(f'Iteration: {u}[')
        u = u+1
        i = 0
        y_pred, acc_y_pred, efficacy_pred = segmentationDir(t, x, y, z, i, temp)

        if y_pred != -1 and acc_y_pred != 0:
            data.append(y_pred)
            media_acc = 0
            media_efficacy = 0
            for e in acc_y_pred:
                media_acc = media_acc + e
            for o in efficacy_pred:
                media_efficacy = media_efficacy + o

            media_acc = media_acc / len(acc_y_pred)
            media_efficacy = media_efficacy / len(efficacy_pred)
            acc.append(media_acc)
            efficacy.append(media_efficacy)
            print(f' -speech: {y_pred}, -accuracy: {media_acc}%, -efficacy: {media_efficacy}%')
            print(f']')

        temp = t[f]/nt
        nt = nt + 1

    M = 0

    iM = 0
    for e in range(0, len(efficacy)):
        if efficacy[e] > M:
            M = efficacy[e]
            iM = e

    return data[iM], acc[iM], M



if __name__ == '__main__':

    print("\nOpen file")

    # ELABORAZIONE DEL FILE
    speeches, accuracy, efficacy = openDir('SegmentationFile', 'frase11.csv')

    # INVIO RISULTATI
    print(f'SPEECH Found :')
    for element in speeches:
        print(f'{element}')

    print(f'Accuracy: {accuracy}%')
    print(f'Efficacy: {efficacy}%')
