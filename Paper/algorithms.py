import math

from Paper import methods_for_input_preprocess, methods_for_diagnosis, methods_for_ranking, functions


def MRSD(instance_num, noa, nof, nor, inum, G, F, T, S):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
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
    print(f'        - number of runs: {nor}')

    # run the algorithm
    diagnoses = methods_for_diagnosis.diagnosis_0(S)

    # rank the diagnoses (diagnoses are normalized!)
    ranked_diagnoses = methods_for_ranking.ranking_0(S, diagnoses)

    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort = functions.calculate_wasted_effort(F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               nor,
               inum,
               str(F),
               'MRSD',
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               wasted_effort]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result

def DMRSD_I1D1R1(instance_num, noa, nof, nor, inum, G, F, T, S):
    """
    :param instance_num: instance number for indexing of experiments
    :param noa: number of agents
    :param nof: number of faulty agents
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
    print(f'        - number of runs: {nor}')

    # prepare inputs: divide the spectrum to local spectra - one for each agent
    local_spectra = methods_for_input_preprocess.input_preprocess_1(noa, S)

    # run the algorithm
    diagnoses = methods_for_diagnosis.diagnosis_1(local_spectra)
    print(f'diagnoses are: {diagnoses}')

    # rank the diagnoses
    ranked_diagnoses = methods_for_ranking.ranking_1(local_spectra, diagnoses)

    # sort the diagnoses according to their rank descending
    ranked_diagnoses.sort(key=lambda diag: diag[1], reverse=True)

    # calculate wasted effort, weighted precision, weighted recall
    wasted_effort = functions.calculate_wasted_effort(F, ranked_diagnoses)
    weighted_precision, weighted_recall = functions.calculate_weighted_precision_and_recall(noa, F, ranked_diagnoses)

    result = [[instance_num,
               noa,
               nof,
               nor,
               inum,
               str(F),
               'DMRSD_I1D1R1',
               '\r\n'.join(list(map(lambda arr: str(arr), diagnoses))),
               '\r\n'.join(list(map(lambda arr: str(arr), ranked_diagnoses))),
               wasted_effort]]
    for k in range(10, 110, 10):
        result[0].append(weighted_precision[math.ceil(len(weighted_precision) * float(k) / 100)-1])
    for k in range(10, 110, 10):
        result[0].append(weighted_recall[math.ceil(len(weighted_recall) * float(k) / 100)-1])

    return result


