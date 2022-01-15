import csv
import random

import flask
import EvolutiveAlgoritm
from multiprocessing import Pool

# Dictionary
speech = {"a", "all", "as", "be", "beau", "best", "birds", "bite", "book", "but", "by", "can", "cant", "co", "cy",
          "day", "der", "do", "done", "dont", "drink", "eye", "fea", "feeds", "fire", "flock", "free", "get", "gether",
          "hand", "have", "him", "hol", "home", "horse", "i", "if", "in", "is", "it", "its", "judge", "lead", "li",
          "like", "lunch", "ma", "make", "mo", "ne", "no", "ny", "o", "of", "off", "place", "po", "pre", "put", "right",
          "rons", "rrow", "self", "sent", "some", "sty", "such", "ter", "that", "the", "ther", "there", "thing", "til",
          "time", "to", "too", "ty", "un", "ver", "wa", "want", "ways", "what", "you", "your"}


def divide_segment(directory, file):
    """Function that divide the input signal into segment of max 10.00 second
       :param directory: directory of the file
       :param file: name of the file
       :return: list of segment of 10 second"""
    t_seg = []
    x_seg = []
    y_seg = []
    z_seg = []
    seg = []

    with open(directory + '/' + file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            if 'TIME' not in line:
                elem = float(line[0])
                t_seg.append(elem)
                elem = float(line[1])
                x_seg.append(elem)
                elem = float(line[2])
                y_seg.append(elem)
                elem = float(line[3])
                z_seg.append(elem)

    stop=10.0
    tmp_x= []
    tmp_y= []
    tmp_z= []
    tmp_t= []
    tmp_seg=[]
    i=0
    while i<len(t_seg):
        if t_seg[i]>=stop:
            stop=stop+10.0
            tmp_seg.append(tmp_t)
            tmp_seg.append(tmp_x)
            tmp_seg.append(tmp_y)
            tmp_seg.append(tmp_z)
            seg.append(tmp_seg)
            tmp_t.clear()
            tmp_x.clear()
            tmp_y.clear()
            tmp_z.clear()

        tmp_t.append(t_seg[i])
        tmp_x.append(x_seg[i])
        tmp_y.append(y_seg[i])
        tmp_z.append(z_seg[i])

        i=i+1

    return seg


def population(t_pop, x_pop, y_pop, z_pop, gene_pop):
    """
      Creation of first solution's population
       :param t_pop: list of time axis
       :param x_pop: list of x axis
       :param y_pop: list of y axis
       :param z_pop: list of z axis
       :param gene_pop: integer that represent the number of persons in the population
       :returns: vectors that represent the first population of segmentation
    """
    print(f'first population: !!!START!!!')
    pop=[]

    for i in range(0, gene_pop):

        tmp_div= random.uniform(0.2, 0.5)
        tmp=0
        person = []
        x_starts = []
        x_ends = []
        fine=-1
        while tmp<=t_pop[len(t_pop) - 1]:
            x_start=tmp
            tmp = tmp + tmp_div
            if tmp <=t_pop[len(t_pop) - 1]:
                x_end=tmp
            else:
                x_end=t_pop[len(t_pop) - 1]

            x_starts.append(x_start)
            x_ends.append(x_end)
            fine=fine+1

        if x_ends[fine]<t_pop[len(t_pop) - 1]:
            x_start = x_ends[fine]
            x_end = t_pop[len(t_pop) - 1]
            x_starts.append(x_start)
            x_ends.append(x_end)

        person.append(x_starts)
        person.append(x_ends)
        pop.append(person)

    print(f'first population: !!!END!!!')

    return pop, t_pop, x_pop, y_pop, z_pop


def f(elem):
    P_first, t, x, y, z = population(elem[0], elem[1], elem[2], elem[3], 500)
    # Elaboration of first population
    P_fit = []
    for p in P_first:
        elem_p = []
        solution, acc, eff = EvolutiveAlgoritm.fit_function(t, x, y, z, p[0], p[1])
        if solution is not None or acc is not None or eff is not None:
            elem_p.append(solution)
            elem_p.append(acc)
            elem_p.append(eff)
            P_fit.append(elem_p)

    print(f'Table: !!!START!!!')
    leader = EvolutiveAlgoritm.selection_sort(P_fit)
    print(f'Table: !!!END!!!')
    # evolution cycle
    leader = EvolutiveAlgoritm.evolution_cycle(t, x, y, z, leader, 500, 50)
    best_solution = leader[0]
    data = best_solution[0]
    accuracy = best_solution[1]
    efficacy = best_solution[2]
    result1 = [data, accuracy, efficacy]

    return result1


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    # Parameter definition
    gene = 500
    epoch = 500
    k = 50
    result = []

    # FILE Reception
    fileCsv = flask.request.files['image']

    print("\nReceived file : " + fileCsv.filename)
    # FILE save
    fileCsv.save('ServerFile/'+fileCsv.filename)
    # FILE elaboration
    segment= divide_segment('ServerFile', fileCsv.filename)
    for elem in segment:
        P_first, t, x, y, z =population(elem[0], elem[1], elem[2], elem[3], gene)
        # Elaboration of first population
        P_fit = []
        for p in P_first:
            elem_p = []
            solution, acc, eff = EvolutiveAlgoritm.fit_function(t, x, y, z, p[0], p[1])
            if solution is not None or acc is not None or eff is not None:
                elem_p.append(solution)
                elem_p.append(acc)
                elem_p.append(eff)
                P_fit.append(elem_p)

        print(f'Table: !!!START!!!')
        leader = EvolutiveAlgoritm.selection_sort(P_fit)
        print(f'Table: !!!END!!!')
        # evolution cycle
        leader = EvolutiveAlgoritm.evolution_cycle(t, x, y, z, leader, epoch, k)
        best_solution = leader[0]
        data = best_solution[0]
        accuracy = best_solution[1]
        efficacy = best_solution[2]
        result1=[data, accuracy, efficacy]
        result.append(result1)

    '''with Pool(processes=len(segment)) as p:
        result.append(p.map(f, segment))'''

    # Sending result
    accuracy_final=0.0
    efficacy_final=0.0
    final_result="phrase: {"
    for element in result:
        accuracy_final=accuracy_final+element[1]
        efficacy_final=efficacy_final+element[2]
        for d in element[0]:
            final_result=final_result+speech[d[2]]+" "

    accuracy_final=accuracy_final/len(result)
    efficacy_final=efficacy_final/len(result)
    final_result=final_result+"}, Accuracy: "+accuracy_final+"%, Efficacy: "+efficacy_final+"%"

    return final_result


if __name__ == '__main__':
    app.run(host="0.0.0.0")
