import flask
import csv
import math
import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf
import EvolutiveAlgoritm

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

def get_spectrogram(waveform):

    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

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


def setElement(file, dir):
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
                elem.append([ math.sqrt(x[n][i]), math.sqrt(y[n][i]), math.sqrt(z[n][i]) ])
            data.append(elem)
            elem = []
        data = np.asarray(data)
        pil_img = tf.keras.preprocessing.image.array_to_img(data)
        data = tf.keras.preprocessing.image.img_to_array(pil_img)
        data = tf.image.resize(data, [224, 224])
        print(f'{dir} {file} : TERMINATO')
    return data


def listDir(directory, file):
    data = []
    x = setElement(file, directory)
    if len(x) > 0:
        data.append(x)
    data = np.asarray(data)
    return data


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():

    #RICEZIONE DEL FILE
    fileCsv = flask.request.files['image']

    print("\nReceived file : " + fileCsv.filename)
    #SALVATAGGIO DEL FILE
    fileCsv.save('ServerFile/'+fileCsv.filename)
    #ELABORAZIONE DEL FILE
    Data = listDir('ServerFile', fileCsv.filename)
    #CARICAMENTO MODELLO
    loaded_model = tf.keras.models.load_model('ReteSpeechModels/Result_Test24')
    y_pred = np.argmax(loaded_model.predict(Data), axis=1)


    #INVIO RISULTATI
    print(f'SPEECH PRONUNCIATA :{y_pred}')
    risultato = f'{y_pred}'

    return risultato


if __name__ == '__main__':
    app.run(host="0.0.0.0")
