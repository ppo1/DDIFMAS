import math

import functions
import methods_for_diagnosis
import methods_for_input_preprocess
import methods_for_ranking

import time

def print_if(verbose, s):
    if verbose:
        print(s) 

# single centralized
def COEF(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running COEF on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, _ = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm
    print_if(verbose, f'COEF:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis = methods_for_diagnosis.diagnosis_coef_0(S, local_spectra)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'COEF:: diagnoses are: {diagnoses}')

    # rank the diagnoses (diagnoses are normalized!) - can choose the step size for the gradient descent
    print_if(verbose, f'COEF:: running ranking')
    t2 = time.time()
    ranked_diagnoses = methods_for_ranking.ranking_coef_0(S, diagnoses)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'COEF:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               'COEF',
               str([len(S) * len(S[0])]),
               str([0]),
               str([0.0]),
               0,
               0.0,
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               0,
               info_sent_diagnosis,
               info_sent_diagnosis * 1.0 / noa,
               str([len(S) * len(S[0])]),
               str([1.0]),
               str([1.0]),
               len(S) * len(S[0]),
               len(S) * len(S[0]),
               len(S) * len(S[0]),
               1.0,
               str([len(S) * len(S[0])]),
               str([1.0]),
               len(S) * len(S[0]),
               1.0,
               len(S) * len(S[0]),
               1.0,
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               -1]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100) - 1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100) - 1])

    return result

# single distributed
def DCOEF_I1D4R2(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False, early_stopping=True):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :param verbose: whether to print the progress of the algorithm
    :param early_stopping: the early stopping parameter
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running DCOEF on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')
    print_if(verbose, f'        - early stopping: {early_stopping}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm
    print_if(verbose, f'DCOEF_I1D4R2:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis = methods_for_diagnosis.diagnosis_4(local_spectra)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'DCOEF_I1D4R2:: diagnoses are: {diagnoses}')

    # rank the diagnoses - collect sent and revealed information in the process
    print_if(verbose, f'DCOEF_I1D4R2:: running ranking')
    t2 = time.time()
    ranked_diagnoses, info_sent_ranking, revealed_information_sum, revealed_information_mean,\
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last, stoped_at = methods_for_ranking.ranking_2(local_spectra, diagnoses, missing_information_cells, nor, early_stopping)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'DCOEF_I1D4R2:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               f"DCOEF_I1D4R2{'_ES' if early_stopping else ''}",
               str(missing_information_cells),
               str([len(S)*len(S[0]) - item for item in missing_information_cells]),
               str([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               max([len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               (info_sent_diagnosis + info_sent_ranking) * 1.0 / noa,
               str(revealed_information_per_agent),
               str([item * 1.0 / (len(S) * len(S[0])) for item in revealed_information_per_agent]),
               str(revealed_information_percent_per_agent),
               revealed_information_sum,
               revealed_information_mean,
               revealed_information_last,
               revealed_information_percent_last,
               str([len(S)*len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in enumerate(missing_information_cells)]),
               str([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item in
                    enumerate(missing_information_cells)]),
               max([len(S)*len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in enumerate(missing_information_cells)]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item in
                    enumerate(missing_information_cells)]),
               max([len(S)*len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in enumerate(missing_information_cells)]) - max([len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in enumerate(missing_information_cells)]) - max([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               stoped_at]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100) - 1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100) - 1])

    return result

# multi centralized
def MRSD(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running MRSD on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, _ = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm
    print_if(verbose, f'MRSD:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis = methods_for_diagnosis.diagnosis_0(S, local_spectra)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'MRSD:: diagnoses are: {diagnoses}')

    # rank the diagnoses (diagnoses are normalized!) - can choose the step size for the gradient descent
    print_if(verbose, f'MRSD:: running ranking')
    t2 = time.time()
    ranked_diagnoses = methods_for_ranking.ranking_0(S, diagnoses, 0.5)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'MRSD:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               'MRSD',
               str([len(S) * len(S[0])]),
               str([0]),
               str([0.0]),
               0,
               0.0,
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               0,
               info_sent_diagnosis,
               info_sent_diagnosis * 1.0 / noa,
               str([len(S) * len(S[0])]),
               str([1.0]),
               str([1.0]),
               len(S) * len(S[0]),
               len(S) * len(S[0]),
               len(S) * len(S[0]),
               1.0,
               str([len(S) * len(S[0])]),
               str([1.0]),
               len(S) * len(S[0]),
               1.0,
               len(S) * len(S[0]),
               1.0,
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               -1]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result

# multi distributed
def DMRSD_I1D1R1(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False, early_stopping=True, alpha=1, huristic_stop=False):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running DMRSD_I1D1R1 on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')
    print_if(verbose, f'        - early stopping: {early_stopping}')
    print_if(verbose, f'        - alpha: {alpha}')
    print_if(verbose, f'        - huristic stop: {huristic_stop}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm, collect diagnosis messages count
    print_if(verbose, f'DMRSD_I1D1R1:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean,\
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last, stoped_component = methods_for_diagnosis.diagnosis_1(local_spectra, missing_information_cells, nor, early_stopping, alpha, huristic_stop)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'DMRSD_I1D1R1:: diagnoses are: {diagnoses}')
    print_if(verbose, f'DMRSD_I1D1R1:: stopped at: {stoped_component}')

    if early_stopping and (alpha < 1 or huristic_stop):
        # remove the unused rows from S and recalculate local spectra
        new_S = [row for row in S if 1 in row and row.index(1) <= stoped_component]
        local_spectra, _ = methods_for_input_preprocess.input_preprocess_1(noa, new_S)

    # rank the diagnoses - can choose the step size for the gradient descent
    print_if(verbose, f'DMRSD_I1D1R1:: running ranking')
    t2 = time.time()
    ranked_diagnoses, info_sent_ranking = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5, stoped_component)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'DMRSD_I1D1R1:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               f'DMRSD_I1D1R1{"_ES" if early_stopping else ""}{"_A" + str(alpha) if alpha != 1 else ""}{"_HS" if huristic_stop else ""}',
               str(missing_information_cells),
               str([len(S) * len(S[0]) - item for item in missing_information_cells]),
               str([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               max([len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               (info_sent_diagnosis + info_sent_ranking) * 1.0 / noa,
               str(revealed_information_per_agent),
               str([item * 1.0 / (len(S) * len(S[0])) for item in revealed_information_per_agent]),
               str(revealed_information_percent_per_agent),
               revealed_information_sum,
               revealed_information_mean,
               revealed_information_last,
               revealed_information_percent_last,
               str([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               str([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]) - max(
                   [len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in enumerate(missing_information_cells)]) - max(
                   [(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               stoped_component]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result

# multi-mc
def DMRSD_I1D2R1(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running DMRSD_I1D2R1 on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm, collect diagnosis messages count
    print_if(verbose, f'DMRSD_I1D2R1:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last = methods_for_diagnosis.diagnosis_2(local_spectra, missing_information_cells)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'DMRSD_I1D2R1:: diagnoses are: {diagnoses}')

    # rank the diagnoses - can choose the step size for the gradient descent
    print_if(verbose, f'DMRSD_I1D2R1:: running ranking')
    t2 = time.time()
    ranked_diagnoses, info_sent_ranking = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'DMRSD_I1D2R1:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               'DMRSD_I1D2R1',
               str(missing_information_cells),
               str([len(S) * len(S[0]) - item for item in missing_information_cells]),
               str([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               max([len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               (info_sent_diagnosis + info_sent_ranking) * 1.0 / noa,
               str(revealed_information_per_agent),
               str([item * 1.0 / (len(S) * len(S[0])) for item in revealed_information_per_agent]),
               str(revealed_information_percent_per_agent),
               revealed_information_sum,
               revealed_information_mean,
               revealed_information_last,
               revealed_information_percent_last,
               str([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               str([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]) - max(
                   [len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in enumerate(missing_information_cells)]) - max(
                   [(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               -1]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100) - 1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100) - 1])

    return result

# multi-smc
def DMRSD_I1D3R1(instance_num, noa, nof, afp, nor, inum, F, S, verbose=False):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param F: faulty agents
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print_if(verbose, f'running DMRSD_I1D3R1 on {instance_num} ({inum}) with:')
    print_if(verbose, f'        - number of agents: {noa}')
    print_if(verbose, f'        - number of faulty agents: {nof}')
    print_if(verbose, f'        - agent fault probability: {afp}')
    print_if(verbose, f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm, collect diagnosis messages count
    print_if(verbose, f'DMRSD_I1D3R1:: running diagnoses')
    t0 = time.time()
    diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last = methods_for_diagnosis.diagnosis_3(local_spectra, missing_information_cells)
    t1 = time.time()
    delta_diag = t1 - t0
    print_if(verbose, f'DMRSD_I1D3R1:: diagnoses are: {diagnoses}')

    # rank the diagnoses - can choose the step size for the gradient descent
    print_if(verbose, f'DMRSD_I1D3R1:: running ranking')
    t2 = time.time()
    ranked_diagnoses, info_sent_ranking = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5)
    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)
    t3 = time.time()
    delta_rank = t3 - t2
    print_if(verbose, f'DMRSD_I1D3R1:: ranked diagnoses are: {ranked_diagnoses}')

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent = \
        functions.calculate_wasted_effort(noa, F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               'DMRSD_I1D3R1',
               str(missing_information_cells),
               str([len(S) * len(S[0]) - item for item in missing_information_cells]),
               str([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               max([len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               (info_sent_diagnosis + info_sent_ranking) * 1.0 / noa,
               str(revealed_information_per_agent),
               str([item * 1.0 / (len(S) * len(S[0])) for item in revealed_information_per_agent]),
               str(revealed_information_percent_per_agent),
               revealed_information_sum,
               revealed_information_mean,
               revealed_information_last,
               revealed_information_percent_last,
               str([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               str([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in
                    enumerate(missing_information_cells)]),
               max([len(S) * len(S[0]) - item + revealed_information_per_agent[ix] for ix, item in
                    enumerate(missing_information_cells)]) - max(
                   [len(S) * len(S[0]) - item for item in missing_information_cells]),
               max([(len(S) * len(S[0]) - item + revealed_information_per_agent[ix]) / (len(S) * len(S[0])) for ix, item
                    in enumerate(missing_information_cells)]) - max(
                   [(len(S) * len(S[0]) - item) / (len(S) * len(S[0])) for item in missing_information_cells]),
               wasted_effort,
               wasted_effort_percent,
               useful_effort,
               useful_effort_percent,
               delta_diag + delta_rank,
               -1]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100) - 1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100) - 1])

    return result
