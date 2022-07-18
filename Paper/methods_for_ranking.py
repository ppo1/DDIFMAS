from Paper import functions
import sympy

def ranking_0(spectrum, diagnoses):
    """
    ranks the diagnoses. for each diagnosis, the diagnoser
    computes a corresponding estimation function
    and then maximizes it
    :param spectrum: the spectrum
    :param diagnoses: the diagnosis list
    :return: ranked diagnosis list
    """
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        print(f'ranking diagnosis: {diagnosis}')

        # divide the spectrum to activity matrix and error vector
        activity_matrix = [row[:-1] for row in spectrum]
        error_vector = [row[-1] for row in spectrum]

        # calculate the probability of the diagnosis
        likelihood = functions.calculate_e_dk(diagnosis, activity_matrix, error_vector)

        # save the result
        ranked_diagnoses.append([diagnosis, likelihood])
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
    # first P calculation
    P = functions.substitute_and_eval(H, LF[0][2])
    # rest P calculations
    for a in list(range(len(LF)))[1:]:
        extended_P = functions.extend_P(P, a, LF[a][0])
        P = functions.substitute_and_eval(H, extended_P)
    return P

def eval_grad(diagnosis, H, P, LF):
    Gradients = {}
    for a in diagnosis:
        rs_function = P / LF[a][2]
        rs_value = functions.substitute_and_eval(H, rs_function)
        gradient_function = rs_value*LF[a][4]
        gradient_value = functions.substitute_and_eval(H, gradient_function)
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients

def update_h(H, Gradients):
    for key in Gradients.keys():
        if H[key] + Gradients[key] > 1.0:
            H[key] = 1.0
        elif H[key] + Gradients[key] < 0.0:
            H[key] = 0.0
        else:
            H[key] = H[key] + Gradients[key]
    return H

def ranking_1(local_spectra, diagnoses):
    """
    ranks the diagnoses. for each diagnosis, the agents
    compute a corresponding partial estimation function
    and then pass numeric results following a certain
    order, until the global function is maximized
    :param local_spectra: the local spectra of each agent
    :param diagnoses: the diagnosis list
    :return: ranked diagnosis list
    """
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to minus infinity
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
        epsilon = 0.005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [0.0]

        # create symbolic local estimation function (LE) for each of the agents
        # create symbolic local derivative function (LD) for each of the agents
        LF, h, r = local_estimation_and_derivative_functions(diagnosis, local_spectra)

        # while true
        while True:
            # calculate P_n(d, H, LS)
            P = eval_P(H, LF)
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
            Gradients = eval_grad(diagnosis, H, P_arr[-1], LF)
            # update H
            H = update_h(H, Gradients)
            # print(P_arr)
            # print(H)

        ranked_diagnoses.append([diagnosis, likelihood])
        print(f'finished ranking diagnosis: {diagnosis}, rank: [{diagnosis},{likelihood}]')
    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses
