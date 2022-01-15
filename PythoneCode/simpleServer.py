import flask
import csv
import math
import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf


def high_pass_filter(x_filter, cutoff):
    """
      High Pass Filter
      it divide the signal of speech from that of movements
      :param x_filter: one of the axis of the data in vector form
      :param cutoff: integer that represent limit of the cutoff
      :return: the data with cutoff application
    """
    fs = len(x_filter)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    f, t, zxx = signal.stft(x_filter)

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
           :return: the spectrogram of signal
        """

    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def interpolate_1d_vector(vector, factor):
    """
        Function of interpolation
         Interpolate, i.e. up sample, a given 1D vector by a specific interpolation factor.
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

    if len(x) >= 50:

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
                elem.append([ math.sqrt(x[n][i]), math.sqrt(y[n][i]), math.sqrt(z[n][i]) ])
            data.append(elem)
            elem = []
        data = np.asarray(data)
        pil_img = tf.keras.preprocessing.image.array_to_img(data)
        data = tf.keras.preprocessing.image.img_to_array(pil_img)
        data = tf.image.resize(data, [224, 224])
        print(f'{dir} {file} : End')
    return data


def list_dir(directory, file):
    """
          Cycle of signal elaboration
           :return: data vector that represent the signal
        """
    
    data = []
    x = set_element(file, directory)
    if len(x) > 0:
        data.append(x)
    data = np.asarray(data)
    return data


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():

    # FILE Reception
    fileCsv = flask.request.files['image']

    print("\nReceived file : " + fileCsv.filename)
    # FILE save
    fileCsv.save('ServerFile/'+fileCsv.filename)
    # FILE elaboration
    Data = list_dir('ServerFile', fileCsv.filename)
    # Model load
    loaded_model = tf.keras.models.load_model('CNN_Models/Result_Test34')
    y_pred = np.argmax(loaded_model.predict(Data), axis=1)

    # Sending result
    print(f'SPEECH PRONUNCIATA :{y_pred}')
    risultato = f'{y_pred}'

    return risultato


if __name__ == '__main__':
    app.run(host="0.0.0.0")
