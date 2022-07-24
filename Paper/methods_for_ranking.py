import copy

from Paper import functions
import sympy

# def ranking_0(spectrum, diagnoses):
#     """
#     ranks the diagnoses. for each diagnosis, the diagnoser
#     computes a corresponding estimation function
#     and then maximizes it
#     :param spectrum: the spectrum
#     :param diagnoses: the diagnosis list
#     :return: ranked diagnosis list
#     """
#     ranked_diagnoses = []
#     for diagnosis in diagnoses:
#         print(f'ranking diagnosis: {diagnosis}')
#
#         # divide the spectrum to activity matrix and error vector
#         activity_matrix = [row[:-1] for row in spectrum]
#         error_vector = [row[-1] for row in spectrum]
#
#         # calculate the probability of the diagnosis
#         likelihood, H = functions.calculate_e_dk(diagnosis, activity_matrix, error_vector)
#
#         # save the result
#         ranked_diagnoses.append([diagnosis, likelihood, H])
#         print(f'finished ranking diagnosis: {diagnosis}, rank: [{diagnosis},{likelihood}]')
#
#     # normalize the diagnosis probabilities
#     normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
#     return normalized_diagnoses

def estimation_and_derivative_functions(diagnosis, spectrum):
    # declare variables
    h = []
    for hj in range(len(spectrum[0][:-1])):
        h.append(sympy.symbols(f'h{hj}'))
    ef, DF = functions.estimation_and_derivative_functions(h, spectrum, diagnosis)
    return ef, DF, h

def eval_grad_R0(diagnosis, H, DF):
    Gradients = {}
    for a in diagnosis:
        gradient_value = functions.substitute_and_eval(H, DF[a])
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients

def ranking_0(spectrum, diagnoses, step):
    """
    ranks the diagnoses. for each diagnosis, the diagnoser
    computes a corresponding estimation function
    and then maximizes it
    :param spectrum: the spectrum
    :param diagnoses: the diagnosis list
    :param step: the gradient step
    :return: ranked diagnosis list
    """
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to zero
        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        # while true
        #   calculate P_n(d, H, M)
        #   if condition is is reached, abort
        #   calculate gradients
        #   update H
        print(f'ranking diagnosis: {diagnosis}')

        # initialize H values of the agents involved in the diagnosis to 0.5
        H = {}
        for a in diagnosis:
            H[f'h{a}'] = 0.5

        # initialize an epsilon value for the stop condition: |P_n(d, H, LS) - P_{n-1}(d, H, LS)| < epsilon
        epsilon = 0.0005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [[0.0, {}]]

        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        ef, DF, h = estimation_and_derivative_functions(diagnosis, spectrum)

        # while true
        while True:
            # calculate P_n(d, H, S)
            P = functions.substitute_and_eval(H, ef)
            P = float(P)
            P_arr.append([P, copy.deepcopy(H)])
            # if condition is reached, abort
            if abs(P_arr[-1][0] - P_arr[-2][0]) < epsilon:
                likelihood = P_arr[-1][0]
                H = P_arr[-1][1]
                break
            if P_arr[-1][0] > 1.0:
                likelihood = P_arr[-2][0]
                H = P_arr[-2][1]
                break
            # calculate gradients
            Gradients = eval_grad_R0(diagnosis, H, DF)
            # update H
            number_of_agents = len(spectrum[0][:-1])
            H, _ = update_h(H, Gradients, step, number_of_agents)
            print(P_arr)
            # print(H)

        ranked_diagnoses.append([diagnosis, likelihood, H])
        print(f'finished ranking diagnosis: {diagnosis}, rank: [{diagnosis},{likelihood}]')

    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses

def local_estimation_and_derivative_functions(diagnosis, local_spectra):
    LF = []
    # declare variables
    h = []
    for hj in range(len(local_spectra)):
        h.append(sympy.symbols(f'h{hj}'))
    r = []
    for ri in range(len(local_spectra[0])):
        r.append(sympy.symbols(f'r{ri}'))
    for a, lsa in enumerate(local_spectra):
        local_table, gpef, lef, gpdf, ldf = functions.local_estimation_and_derivative_functions_for_agent(h, r, a, lsa, diagnosis)
        LF.append([local_table, gpef, lef, gpdf, ldf])
    return LF, h, r

def eval_P(H, LF):
    # information sent during the evaluation of P
    information_sent_eval_P = 0
    # first P calculation
    P = functions.substitute_and_eval(H, LF[0][2])
    # rest P calculations
    for a in list(range(len(LF)))[1:]:
        information_sent_eval_P += 1
        extended_P = functions.extend_P(P, a, LF[a][0])
        P = functions.substitute_and_eval(H, extended_P)
    return P, information_sent_eval_P

def eval_grad(diagnosis, H, P, LF):
    information_sent_eval_grad = 0
    Gradients = {}
    for a in diagnosis:
        information_sent_eval_grad += 1
        rs_function = P / LF[a][2]
        rs_value = functions.substitute_and_eval(H, rs_function)
        gradient_function = rs_value*LF[a][4]
        gradient_value = functions.substitute_and_eval(H, gradient_function)
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients, information_sent_eval_grad

def update_h(H, Gradients, step, number_of_agents):
    information_sent_update_h = 0
    for key in Gradients.keys():
        information_sent_update_h += number_of_agents - 1
        if H[key] + step * Gradients[key] > 1.0:
            H[key] = 1.0
        elif H[key] + step * Gradients[key] < 0.0:
            H[key] = 0.0
        else:
            H[key] = H[key] + step * Gradients[key]
    return H, information_sent_update_h

def ranking_1(local_spectra, diagnoses, step):
    """
    ranks the diagnoses. for each diagnosis, the agents
    compute a corresponding partial estimation function
    and then pass numeric results following a certain
    order, until the global function is maximized
    :param local_spectra: the local spectra of each agent
    :param diagnoses: the diagnosis list
    :param step: the gradient step
    :return: ranked diagnosis list
    """
    information_sent = 0
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to zero
        # create symbolic local estimation function (LE) for each of the agents
        # create symbolic local derivative function (LD) for each of the agents
        # while true
        #   calculate P_n(d, H, M)
        #   if condition is is reached, abort
        #   calculate gradients
        #   update H
        print(f'ranking diagnosis: {diagnosis}')

        # initialize H values of the agents involved in the diagnosis to 0.5
        H = {}
        for a in diagnosis:
            H[f'h{a}'] = 0.5

        # initialize an epsilon value for the stop condition: |P_n(d, H, LS) - P_{n-1}(d, H, LS)| < epsilon
        epsilon = 0.0005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [0.0]

        # create symbolic local estimation function (LE) for each of the agents
        # create symbolic local derivative function (LD) for each of the agents
        LF, h, r = local_estimation_and_derivative_functions(diagnosis, local_spectra)

        # while true
        while True:
            # calculate P_n(d, H, LS) and record the sent information
            P, information_sent_eval_P = eval_P(H, LF)
            information_sent += information_sent_eval_P
            P = float(P)
            P_arr.append(P)
            # if condition is is reached, abort
            if abs(P_arr[-1] - P_arr[-2]) < epsilon:
                likelihood = P_arr[-1]
                break
            if P_arr[-1] > 1.0:
                likelihood = P_arr[-2]
                break
            # calculate gradients
            Gradients, information_sent_eval_grad = eval_grad(diagnosis, H, P_arr[-1], LF)
            information_sent += information_sent_eval_grad
            # update H
            number_of_agents = len(local_spectra)
            H, information_sent_update_h = update_h(H, Gradients, step, number_of_agents)
            information_sent += information_sent_update_h
            # print(P_arr)
            # print(H)

        ranked_diagnoses.append([diagnosis, likelihood, H])
        print(f'finished ranking diagnosis: {diagnosis}, rank: [{diagnosis},{likelihood}]')
    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses, information_sent
