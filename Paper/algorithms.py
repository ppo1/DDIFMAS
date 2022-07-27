import math

from Paper import methods_for_input_preprocess, methods_for_diagnosis, methods_for_ranking, functions


def MRSD(instance_num, noa, nof, afp, nor, inum, G, F, T, S):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param G: graph
    :param F: faulty agents
    :param T: traces
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print(f'running MRSD on {instance_num} ({inum}) with:')
    print(f'        - number of agents: {noa}')
    print(f'        - number of faulty agents: {nof}')
    print(f'        - agent fault probability: {afp}')
    print(f'        - number of runs: {nor}')

    # run the algorithm
    diagnoses = methods_for_diagnosis.diagnosis_0(S)

    # rank the diagnoses (diagnoses are normalized!) - can choose the step size for the gradient descent
    ranked_diagnoses = methods_for_ranking.ranking_0(S, diagnoses, 0.5)

    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort = functions.calculate_wasted_effort(F, ranked_diagnoses)
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
               -1,
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               -1,
               -1,
               -1,
               -11,
               -11,
               -11,
               -11,
               -11,
               -11,
               wasted_effort]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result

def DMRSD_I1D1R1(instance_num, noa, nof, afp, nor, inum, G, F, T, S):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param G: graph
    :param F: faulty agents
    :param T: traces
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print(f'running DMRSD_I1D1R1 on {instance_num} ({inum}) with:')
    print(f'        - number of agents: {noa}')
    print(f'        - number of faulty agents: {nof}')
    print(f'        - agent fault probability: {afp}')
    print(f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm, collect diagnosis messages count
    diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean,\
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last = methods_for_diagnosis.diagnosis_1(local_spectra, missing_information_cells)
    print(f'diagnoses are: {diagnoses}')

    # rank the diagnoses - can choose the step size for the gradient descent
    ranked_diagnoses, info_sent_ranking = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5)

    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort = functions.calculate_wasted_effort(F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               afp,
               nor,
               inum,
               str(F),
               '\r\n'.join(list(map(lambda arr: str(arr), S))),
               'DMRSD_I1D1R1',
               str(missing_information_cells),
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               revealed_information_sum,
               revealed_information_mean,
               str(revealed_information_per_agent),
               revealed_information_last,
               str(revealed_information_percent_per_agent),
               revealed_information_percent_last,
               wasted_effort]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result

def DMRSD_I1D2R1(instance_num, noa, nof, afp, nor, inum, G, F, T, S):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
    :param afp: agent fault probability
    :param nor: number of runs
    :param inum: instance number
    :param G: graph
    :param F: faulty agents
    :param T: traces
    :param S: spectrum
    :return:
    """
    # announcing instance parameters
    print(f'running DMRSD_I1D2R1 on {instance_num} ({inum}) with:')
    print(f'        - number of agents: {noa}')
    print(f'        - number of faulty agents: {nof}')
    print(f'        - agent fault probability: {afp}')
    print(f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra, missing_information_cells = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm, collect diagnosis messages count
    diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last = methods_for_diagnosis.diagnosis_2(local_spectra, missing_information_cells)
    print(f'diagnoses are: {diagnoses}')

    # rank the diagnoses - can choose the step size for the gradient descent
    ranked_diagnoses, info_sent_ranking = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5)

    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort = functions.calculate_wasted_effort(F, ranked_diagnoses)
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
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               len(ranked_diagnoses),
               info_sent_diagnosis,
               info_sent_ranking,
               info_sent_diagnosis + info_sent_ranking,
               revealed_information_sum,
               revealed_information_mean,
               str(revealed_information_per_agent),
               revealed_information_last,
               str(revealed_information_percent_per_agent),
               revealed_information_percent_last,
               wasted_effort]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100) - 1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100) - 1])

    return result
