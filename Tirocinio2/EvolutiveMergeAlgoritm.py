import csv
import math

import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf


# FILTRO PASSA ALTO
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
def set_element(x, y, z):
    x = x
    y = y
    z = z
    data = []

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
                elem.append([math.sqrt(x[n][i]), math.sqrt(y[n][i]), math.sqrt(z[n][i])])
            data.append(elem)
            elem = []
        data = np.asarray(data)
        pil_img = tf.keras.preprocessing.image.array_to_img(data)
        data = tf.keras.preprocessing.image.img_to_array(pil_img)
        data = tf.image.resize(data, [224, 224])
        return data
    else:
        return None


def list_dir(x, y, z):
    data = []
    w = set_element(x, y, z)
    if w is not None:
        if len(w) > 0:
            data.append(w)
        data = np.asarray(data)
        return data
    else:
        return None


def perc_prediction(pred):
    e_x = np.exp(pred - np.max(pred))
    return e_x / e_x.sum()


def cromosoma(t, x, y, z, seg_start, seg_end):
    x1 = []
    y1 = []
    z1 = []
    tmp =0
    i = 0
    for e in range(0, len(t)):
        if t[e]==seg_start:
            i=e

    while i < len(x) and t[i] <= seg_end:
        x1.append(x[i])
        y1.append(y[i])
        z1.append(z[i])
        i = i+1

    speech = list_dir(x1, y1, z1)
    if speech is not None:
        loaded_model = tf.keras.models.load_model('ReteSpeechModels/Result_Test26')
        prediction = loaded_model.predict(speech)
        y_pred = np.argmax(prediction)
        prediction_perc = perc_prediction(prediction)
        acc_y_pred = prediction_perc[0][y_pred]*100
        prediction_media = 0.0
        for e in prediction_perc[0]:
            if e != prediction_perc[0][y_pred]:
                prediction_media = prediction_media + e

        prediction_media = prediction_media / (len(prediction_perc[0]) - 1)
        efficacy_pred = acc_y_pred - (prediction_media * 100)

        return y_pred, acc_y_pred, efficacy_pred
    else:
        return -1, 0, 0


def fit_function(t, x, y, z, x_start, x_end):
    print(f'Search Speech')
    if np.size(x_start)>0 and np.size(x_end)>0:
        fitness= np.empty(shape=(len(x_start), 5), dtype='object')
        elem=0
        while elem <len(x_start):
            y_pred, acc_y_pred, efficacy_pred = cromosoma(t, x, y, z, x_start[elem], x_end[elem])
            fitness[elem][0]= x_start[elem]
            fitness[elem][1]= x_end[elem]
            fitness[elem][2]= y_pred
            fitness[elem][3]= acc_y_pred
            fitness[elem][4]= efficacy_pred
            elem=elem+1

        media_acc = 0
        media_efficacy = 0
        for e in fitness:
            acc= e[3]
            eff= e[4]
            media_acc = media_acc + acc
            media_efficacy = media_efficacy + eff

        media_acc = media_acc / len(fitness)
        media_efficacy = media_efficacy / len(fitness)
        print(f'fitness: {fitness}, -accuracy: {int(media_acc)}%, -efficacy:{int(media_efficacy)}%')

        return fitness, media_acc, media_efficacy
    else:
        return None, None, None


def population(directory, file):
    print(f'first population: !!!START!!!')
    t = []
    x = []
    y = []
    z = []
    pop=[]

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

    x_start=t[0]
    x_end=0
    x_starts = []
    x_ends = []
    p = []
    for e in range(0, len(x)):
        if x[e]==0.08 and y[e]==-0.505 and 9.80 < z[e] < 10.1:
            x_end=t[e-1]
            x_starts.append(x_start)
            x_ends.append(x_end)
            x_start=t[e]
    p.append(x_starts)
    p.append(x_ends)
    pop.append(p)

    print(f'first population: !!!END!!!')

    return pop, t, x, y, z


def crossover(p):
    print(f'Crossover: !!!START!!!')
    p1= []
    x_starts= []
    x_ends= []
    data =[]
    for e in p:
        data.append(e)

    cont = 0
    while cont < len(data):
        if cont + 1 < len(data):
            s1=data[cont]
            s2=data[cont+1]
            x_starts.append(s1[0])
            x_ends.append(s2[1])
            cont=cont+2
        else:
            s = data[cont]
            x_starts.append(s[0])
            x_ends.append(s[1])
            cont = cont + 2

    p1.append(x_starts)
    p1.append(x_ends)
    print(f'Crossover: !!!END!!!')

    return p1


def mutation(old, new):
    b=[]
    cont=0

    for elem in range(0, len(new)):
        if cont+1<len(old):
            elem_new=new[elem]
            elem_old1=old[cont]
            elem_old2=old[cont+1]
            media_acc_old=(elem_old1[2]+elem_old2[2])/2
            if elem_new[2]>=media_acc_old:
                b.append(elem_new)
            else:
                b.append(elem_old1)
                b.append(elem_old2)

            cont=cont+2
        else:
            elem_new = new[elem]
            b.append(elem_new)
            cont = cont + 2

    return b

if __name__ == '__main__':

    print("\nOpen file")
    # ELABORAZIONE DEL FILE
    P_first, t, x, y, z = population('SegmentationFile', 'frase1.csv')
    print(f'first population: {P_first}')

    # Elaborazione risultati prima popolazione
    P_fit=[]
    p=P_first[0]
    elem_p=[]
    solution, acc, eff= fit_function(t, x, y, z, p[0], p[1])
    if solution is not None or acc is not None or eff is not None:
        elem_p.append(solution)
        elem_p.append(acc)
        elem_p.append(eff)
        P_fit.append(elem_p)

    P_cross=crossover(P_fit[0])
    P_fit_cross = []
    elem_p = []
    solution, acc, eff = fit_function(t, x, y, z, P_cross[0], P_cross[1])
    if solution is not None or acc is not None or eff is not None:
        elem_p.append(solution)
        elem_p.append(acc)
        elem_p.append(eff)
        P_fit_cross.append(elem_p)

    P_mut=mutation(P_fit[0], P_fit_cross[0])

    fine=False
    while not fine:
        P_prec=P_fit[0]
        if len(P_mut)==len(P_prec):
            fine=True
        else:
            P_fit=P_mut
            P_cross = crossover(P_fit)
            P_fit_cross = []
            elem_p = []
            solution, acc, eff = fit_function(t, x, y, z, P_cross[0], P_cross[1])
            if solution is not None or acc is not None or eff is not None:
                elem_p.append(solution)
                elem_p.append(acc)
                elem_p.append(eff)
                P_fit_cross.append(elem_p)

            P_mut = mutation(P_fit, P_fit_cross[0])


# INVIO RISULTATI
    media_acc = 0
    media_eff = 0
    print(f'Speech found:')
    for elem in P_mut:
        print(f'{elem[2]}')
        media_acc = media_acc + elem[3]
        media_eff = media_eff + elem[4]

    media_acc = media_acc / len(P_mut)
    media_eff = media_eff / len(P_mut)

    print(f'Accuracy: {media_acc}%')
    print(f'Efficacy: {media_eff}%')
