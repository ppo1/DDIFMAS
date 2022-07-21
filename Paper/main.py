import random
import matplotlib.pyplot as plt
import xlsxwriter
import networkx as nx
from Paper import algorithms
import numpy as np
from random import randrange

def create_random_graph(noa):
    # use erdos-renyi algorithm to generate random graph
    G = nx.erdos_renyi_graph(noa, 0.5, seed=123, directed=False)
    # make sure it is connected graph
    connected = nx.is_connected(G)
    # while the graph is not connected, generate a new one
    while not connected:
        G = nx.erdos_renyi_graph(noa, 0.5, seed=123, directed=False)
        connected = nx.is_connected(G)
    # plot the graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    return G

def choose_faulty_agents(noa_l, nof):
    # generate an ordered list of the agents (their numbers)
    F = [i for i in range(noa_l)]
    # shuffle the list
    random.shuffle(F)
    # choose the first number of faulty agents from the list to be faulty
    F = F[:nof]
    return F

def adjacency_matrix(A):
    # build an adjacency list for each node
    res_A = [[] for _ in A]
    # for each node (i) in the input matrix A
    for i, row in enumerate(A):
        # for each node (c) in that row
        for j, c in enumerate(row):
            # if the node (c) equals one, that means it is a neighbour of node (i)
            if c == 1:
                res_A[i].append(j)
    return res_A

def generate_traces(G, noa, nor):
    T = []
    # let networkx return the adjacency matrix A
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)
    A = adjacency_matrix(A)
    for i in range(nor):
        # choose a random starting node
        current_node = randrange(noa)
        # choose a random trace length
        trace_length = 2 + randrange(noa)
        # initialize the trace
        trace = [current_node]
        # while can advance (neighbours not visited previously)
        while len(trace) < trace_length:
            available_neighbours = [i for i in A[current_node] if i not in trace]
            if len(available_neighbours) == 0:
                break
            # choose random neighbour and advance there
            random.shuffle(available_neighbours)
            current_node = available_neighbours[0]
            trace.append(current_node)
        # append the resulting trace
        T.append(trace)
    return T

def generate_spectrum(noa, nor, F, T):
    # create an empty (with 2's) spectrum
    S = [[2 for _ in range(noa + 1)] for _ in range(nor)]

    # for each run (row) in the spectrum
    for i, run in enumerate(S):
        # initialize the error vector cell to 0
        run[-1] = 0
        # for each agent in that row
        for j in range(len(run[:-1])):
            # if it was in the corresponding trace update its cell to 1
            if j in T[i]:
                run[j] = 1
                # if it is in the faulty agents list update the error vector cell to 1
                if j in F:
                    run[-1] = 1
            else:
                run[j] = 0

    return S

def write_data_to_excel(data):
    columns = [
        {'header': 'instance_number'},
        {'header': 'noa'},
        {'header': 'nof'},
        {'header': 'nor'},
        {'header': 'noi'},
        {'header': 'oracle'},
        {'header': 'spectrum'},
        {'header': 'diagnosis algorithm'},
        {'header': 'diagnoses'},
        {'header': 'ranked diagnoses'},
        {'header': 'number of diagnoses'},
        {'header': 'Information Sent - Diagnosis'},
        {'header': 'Information Sent - Ranking'},
        {'header': 'Information Sent - Both'},
        {'header': 'Wasted Effort'},
        {'header': 'Weighted Precision 10'},
        {'header': 'Weighted Precision 20'},
        {'header': 'Weighted Precision 30'},
        {'header': 'Weighted Precision 40'},
        {'header': 'Weighted Precision 50'},
        {'header': 'Weighted Precision 60'},
        {'header': 'Weighted Precision 70'},
        {'header': 'Weighted Precision 80'},
        {'header': 'Weighted Precision 90'},
        {'header': 'Weighted Precision 100'},
        {'header': 'Weighted Recall 10'},
        {'header': 'Weighted Recall 20'},
        {'header': 'Weighted Recall 30'},
        {'header': 'Weighted Recall 40'},
        {'header': 'Weighted Recall 50'},
        {'header': 'Weighted Recall 60'},
        {'header': 'Weighted Recall 70'},
        {'header': 'Weighted Recall 80'},
        {'header': 'Weighted Recall 90'},
        {'header': 'Weighted Recall 100'}
    ]
    # write the data to xlsx file
    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(data), len(columns) - 1, {'data': data, 'columns': columns})
    workbook.close()

def run_random_experiments(number_of_agents, number_of_faulty, number_of_runs, number_of_instances):
    results = []
    noa_l = len(number_of_agents)
    nof_l = len(number_of_faulty)
    nor_l = len(number_of_runs)
    noi_l = number_of_instances
    total_instances = noa_l * nof_l * nor_l * noi_l
    for noa_i, noa in enumerate(number_of_agents):
        G = create_random_graph(noa)
        for nof_i, nof in enumerate(number_of_faulty):
            F = choose_faulty_agents(noa_l, nof)
            F.sort()
            for nor_i, nor in enumerate(number_of_runs):
                for inum in range(number_of_instances):
                    instance_num = noa_i * (nof_l * nor_l * noi_l) + nof_i * (nor_l * noi_l) + nor_i * noi_l + inum + 1
                    T = generate_traces(G, noa, nor)
                    S = generate_spectrum(noa, nor, F, T)
                    print(f'running instance {instance_num}/{total_instances} ({inum+1}/{number_of_instances}) with:')
                    print(f'        - number of agents: {noa} ({noa_i+1})')
                    print(f'        - number of faulty agents: {nof} ({nof_i+1})')
                    print(f'        - number of runs: {nor} ({nor_i+1})')
                    result_mrsd = algorithms.MRSD(instance_num, noa, nof, nor, inum + 1, G, F, T, S)
                    result_dmrsd1 = algorithms.DMRSD_I1D1R1(instance_num, noa, nof, nor, inum + 1, G, F, T, S)
                    results += result_mrsd
                    results += result_dmrsd1
    write_data_to_excel(results)
    print(9)


if __name__ == '__main__':
    print('Hi, PyCharm')

    # run_random_experiments([5, 6, 7, 8, 9], [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 10)
    run_random_experiments([7, 8, 9], [2], [10], 10)

    print('Bye, PyCharm')
