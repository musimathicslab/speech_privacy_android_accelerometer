import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy import signal
import tensorflow as tf


def high_pass_filter(x, cutoff):
    """
      FILTRO PASSA ALTO
      it divide the signal of speech from that of movements
      :param x: one of the axis of the data in vector form
      :param cutoff: integer that represent limit of the cutoff
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
       :return: the spectrogram of signal
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


def set_element(x, y, z):
    """
       Signal load and pre-elaboration
       :param x: vector of x axis
       :param y: vector of y axis
       :param z: vector of z axis
       :return: data vector that represent the signal
    """
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
    """
       Cycle of signal elaboration
       :param x: vector of x axis
       :param y: vector of y axis
       :param z: vector of z axis
       :return: data vector that represent the signal
    """
    data = []
    w = set_element(x, y, z)
    if w is not None:
        if len(w) > 0:
            data.append(w)
        data = np.asarray(data)
        return data
    else:
        return None


# extrapolation of percent data
def perc_prediction(pred):
    e_x = np.exp(pred - np.max(pred))
    return e_x / e_x.sum()


def cromosoma(t, x, y, z, seg_start, seg_end):
    """
      Search for a speech and accuracy assignment
       :param x: vector of x axis
       :param y: vector of y axis
       :param z: vector of z axis
       :param t: vector of time axis
       :param seg_start: integer tha represent the start of signal segment
       :param seg_end: integer tha represent the end of signal segment
       :returns: speeches, accuracy, efficacy of segment
    """
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
        loaded_model = tf.keras.models.load_model('ReteSpeechModels/Result_Test34')
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
    """
       Definition of the efficiency of a solution
       :param x: vector of x axis
       :param y: vector of y axis
       :param z: vector of z axis
       :param t: vector of time axis
       :param x_start : integer tha represent the start of signal segment
       :param x_end: integer tha represent the end of signal segment
       :returns: speeches, accuracy, efficacy of all the signals solution
    """
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


def population(directory, file, gene):
    """
      Creation of first solution's population
       :param directory: dir of a file
       :param file: name of the file
       :param gene: integer that represent the number of persons in the population
       :returns: vectors that represent the first population of segmentation
    """
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

    for i in range(0, gene):

        tmp_div= random.uniform(0.2, 0.5)
        tmp=0
        p = []
        x_starts = []
        x_ends = []
        fine=-1
        while tmp<=t[len(t)-1]:
            x_start=tmp
            tmp = tmp + tmp_div
            if tmp <=t[len(t)-1]:
                x_end=tmp
            else:
                x_end=t[len(t)-1]

            x_starts.append(x_start)
            x_ends.append(x_end)
            fine=fine+1

        if x_ends[fine]<t[len(t)-1]:
            x_start = x_ends[fine]
            x_end = t[len(t)-1]
            x_starts.append(x_start)
            x_ends.append(x_end)

        p.append(x_starts)
        p.append(x_ends)
        pop.append(p)

    print(f'first population: !!!END!!!')

    return pop, t, x, y, z


def selection_sort(a):
    """
       Generate a Table of best K solution in growing order
       :param a: vector of the solutions
       :return: vector in growing order
    """
    v = []
    for elem in a:
        v.append(elem)

    for ind in range(0, len(v)):
        massimo = ind
        for M in range(ind+1, len(v)):
            s1=v[M]
            s2=v[massimo]
            if s1[2] >= s2[2]:
                massimo = M

        temporary=v[ind]
        v[ind]=v[massimo]
        v[massimo]=temporary

    return v


def selection_sort2(a):
    """
       Generate a Table of best K solution in decreasing order
       :param a: vector of the solutions
       :return: vector in decreasing order
    """
    v = []
    for elem in a:
        v.append(elem)

    for ind in range(0, len(v)):
        minimo = ind
        for M in range(ind+1, len(v)):
            s1=v[M]
            s2=v[minimo]
            if len(s1[0]) <= len(s2[0]):
                minimo = M

        temporary=v[ind]
        v[ind]=v[minimo]
        v[minimo]=temporary

    return v


def crossover(p1, p2):
    """
    Function that took two consecutive people of the population, divides each person's solution into two halves and
      combine half of one with half of the other and the other way around
      :param p1: vector that represent a person population 
      :param p2: vector that represent a person population 
      :returns: the two person merged each other
    """
    print(f'Crossover: !!!START!!!')
    pcross1 = []
    pcross2 = []
    data1 = []
    data2 = []
    for e in p1:
        data1.append(e)
    for e in p2:
        data2.append(e)

    s1_last=data1[len(data1)-1]
    t1=s1_last[1]
    t1=t1/2
    s2_last = data2[len(data2) - 1]
    t2 = s2_last[1]
    t2=t2/2

    x_starts = []
    x_ends = []
    x_starts1 = []
    x_ends1 = []
    x_starts2 = []
    x_ends2 = []
    x_starts3 = []
    x_ends3 = []
    for seg1 in data1:
        x_start = seg1[0]
        x_end = seg1[1]
        if x_end<=t1:
            x_starts.append(x_start)
            x_ends.append(x_end)
        else:
            x_starts1.append(x_start)
            x_ends1.append(x_end)

    for seg2 in data2:
        x_start = seg2[0]
        x_end = seg2[1]
        if x_end<=t2:
            x_starts2.append(x_start)
            x_ends2.append(x_end)
        else:
            x_starts3.append(x_start)
            x_ends3.append(x_end)

    for el in x_starts3:
        x_starts.append(el)
    for el1 in x_ends3:
        x_ends.append(el1)
    for el3 in x_starts1:
        x_starts2.append(el3)
    for el4 in x_ends1:
        x_ends2.append(el4)

    pcross1.append(x_starts)
    pcross1.append(x_ends)
    pcross2.append(x_starts2)
    pcross2.append(x_ends2)

    print(f'Crossover: !!!END!!!')

    return pcross1, pcross2


def mutation_division(p):
    """
       Function that divide a segment into a random segment's point
       :param p: vector that represent a person of the solution's population
       :return: vector of the changed person
    """
    print(f'Mutation division: !!!START!!!')
    p1= []
    x_starts= []
    x_ends= []
    data =[]
    for e in p:
        data.append(e)

    for seg in data:
        x_start= seg[0]
        x_end= seg[1]
        if x_end-x_start>=0.4:
            d= random.uniform(x_start+0.2, x_end-0.2)

            x_starts.append(x_start)
            x_starts.append(d)
            x_ends.append(d)
            x_ends.append(x_end)
        else:
            x_starts.append(x_start)
            x_ends.append(x_end)

    p1.append(x_starts)
    p1.append(x_ends)
    print(f'Mutation Division: !!!END!!!')

    return p1


def mutation_join(p):
    """ 
       Function that join two segment in one
        :param p: vector that represent a person of the population
        :return: changed person
    """
    print(f'Mutaion Join: !!!START!!!')
    p1= []
    x_starts= []
    x_ends= []
    data =[]
    for e in p:
        data.append(e)

    cont=0
    while cont<len(data):
        if cont+1<len(data):
            seg1=data[cont]
            seg2=data[cont+1]
            x_start= seg1[0]
            x_end= seg2[1]
            if (x_end-x_start)<2.0:
                x_starts.append(x_start)
                x_ends.append(x_end)
            else:
                seg1 = data[cont]
                x_start = seg1[0]
                x_end = seg1[1]
                x_starts.append(x_start)
                x_ends.append(x_end)
        else:
            seg1 = data[cont]
            x_start = seg1[0]
            x_end = seg1[1]
            x_starts.append(x_start)
            x_ends.append(x_end)

        cont=cont+2

    p1.append(x_starts)
    p1.append(x_ends)
    print(f'Mutation Join: !!!END!!!')

    return p1


def mutation_random(p):
    """ 
       Function that create a mutation of a segment
       Mutation:- left shift
              - right shift
              - join
              -division
       :param p: vector that represent the person of the solution's population
       :return: changed person
    """
    print(f'Mutation: !!!START!!!')
    p1= []
    x_starts = []
    x_ends = []
    data = []
    for e in p:
        data.append(e)

    elem=0
    while elem<len(data):
        fitness=data[elem]
        x_start= fitness[0]
        x_end= fitness[1]
        mut = random.randint(0, 2)
        if mut == 0:
            # first mutation
            # a segment is shifted on the right or on the left
            # of the original signal
            shift= random.randint(0, 1)
            if shift==0:
                print(f'Mutation: shift SX')
                if elem-1>0:
                    if x_start>0.2:
                        x_start2=x_start-0.2
                        ind_last=len(x_ends)-1
                        x_ends[ind_last]= x_ends[ind_last] - 0.2
                        x_starts.append(x_start2)
                        x_ends.append(x_end)
                    else:
                        x_starts.append(x_start)
                        x_ends.append(x_end)
                else:
                    x_starts.append(x_start)
                    x_ends.append(x_end)
            if shift==1:
                print(f'Mutation: shift DX')
                if elem+1<len(data):
                    if x_end>t[len(t)-1]+0.2:
                        x_end2= x_end+0.2
                        fitness = data[elem + 1]
                        fitness[0]=fitness[0]+0.2
                        data[elem + 1]= fitness
                        x_starts=x_start
                        x_ends.append(x_end2)
                    else:
                        x_starts.append(x_start)
                        x_ends.append(x_end)
                else:
                    x_starts.append(x_start)
                    x_ends.append(x_end)
        if mut == 1:
            # second mutation
            # the two result segments were created by
            # a division in a random point of the starting segment
            print(f'Mutation: division')
            c=(x_end-x_start)/2
            if c>0.1:
                x_starts.append(x_start)
                x_starts.append(x_start+c)
                x_ends.append(x_start+c)
                x_ends.append(x_end)
            else:
                x_starts.append(x_start)
                x_ends.append(x_end)

        if mut == 2:
            # third mutation
            # the result segment is a join between
            # two consecutive segments
            print(f'Mutation: join')
            x_starts.append(x_start)
            if elem + 1 < len(data):
                fitness1 = data[elem + 1]
                x_end1 = fitness1[1]
                x_ends.append(x_end1)
                elem= elem + 1
            else:
                x_ends.append(x_end)

        elem= elem + 1

    p1.append(x_starts)
    p1.append(x_ends)
    print(f'Mutation: !!!END!!!')

    return p1


def generate_graphic(a, msg):
    """
      Function that generate a graphic which represent the behaviour of accuracy
       :param a: vector that represent the population result of the segmentation
       :param msg: message to print
    """
    x_axis = []
    y_axis = []
    y1_axis = []
    P_graphic = selection_sort2(a)
    for elem in P_graphic:
        x_axis.append(len(elem[0]))
        y_axis.append(elem[1])
        y1_axis.append(elem[2])

    plt.axis([0, x_axis[len(x_axis) - 1], 0, 100])
    plt.grid()
    plt.xticks([1 * k for k in range(0, len(x_axis))])
    plt.yticks([10 * j for j in range(0, 10)])
    fr='trend of accuracy and efficacy '+msg
    plt.title(fr)
    plt.xlabel('speech division number')
    plt.ylabel('% of')
    plt.plot(x_axis, y_axis, color='red', label='accuracy')
    plt.plot(x_axis, y1_axis, color='blue', label='efficacy')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    print("\nOpen file")

    # Parameter definition
    gene=500
    epoch=100
    k=30

    # Elaboration of FILE and building of the first population
    P_first, t, x, y, z = population('SegmentationFile', 'frase12.csv', gene)
    print(f'first population: {P_first}')

    # Elaboration of first population
    P_fit=[]
    for p in P_first:
        elem_p=[]
        solution, acc, eff= fit_function(t, x, y, z, p[0], p[1])
        if solution is not None or acc is not None or eff is not None:
            elem_p.append(solution)
            elem_p.append(acc)
            elem_p.append(eff)
            P_fit.append(elem_p)

    # result of first population
    generate_graphic(P_fit, 'of first population')

    print(f'Table: !!!START!!!')
    leader=selection_sort(P_fit)
    print(f'Table: !!!END!!!')

    # Evolution Cycle
    for i in range(0, epoch):
        if i==25 or i==50:
            s= 'after some iterations'
            generate_graphic(leader, s)
        print(f'EPOCH: {i+1}')
        P_new = []
        best_P=[]
        for j in range(0, k):
            best_P.append(leader[j])
            print(f'Position: {j}-->')
            print(f'Population: {leader[j]}<--')

        u=0
        for i in range(0, len(best_P)-1):
            u = u + 1
            print(f'Evolution: {u}')
            sol=best_P[i]

            if i<len(best_P)-1:
                sol1=best_P[i+1]
                p_cross, p_cross1=crossover(sol[0], sol1[0])
            else:
                sol1=best_P[0]
                p_cross, p_cross1 = crossover(sol[0], sol1[0])

            p_div= mutation_division(sol[0])
            # p_join= mutation_join(sol[0])
            p_mutation=mutation_random(sol[0])
            P_old=[]
            end_old=[]
            start_old=[]
            for e in sol[0]:
                start_old.append(e[0])
                end_old.append(e[1])

            P_old.append(start_old)
            P_old.append(end_old)
            P_new.append(P_old)
            P_new.append(p_cross)
            P_new.append(p_cross1)
            P_new.append(p_div)
            # P_new.append(p_join)
            P_new.append(p_mutation)

        P_fit_new = []
        for p in P_new:
            elem_p = []
            solution, acc, eff = fit_function(t, x, y, z, p[0], p[1])
            if solution is not None or acc is not None or eff is not None:
                elem_p.append(solution)
                elem_p.append(acc)
                elem_p.append(eff)
                P_fit_new.append(elem_p)

        print(f'Table: !!!START!!!')
        leader = selection_sort(P_fit_new)
        print(f'Table: !!!END!!!')

    generate_graphic(leader, 'at the end of solution')

    # Sending result
    best_solution=leader[0]
    data = best_solution[0]
    accuracy=best_solution[1]
    efficacy=best_solution[2]
    print(f'Speech found:')
    for element in data:
        print(f'{element[2]}')
    print(f'-Accuracy: {accuracy}%')
    print(f'-Efficacy: {efficacy}%')
