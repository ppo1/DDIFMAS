from Paper import functions

def diagnosis_0(spectrum):
    """
    the traditional hitting set algorithm
    :param spectrum: the local spectra of the agents
    :return: a set of diagnoses
    """
    # calculate conflicts
    conflicts = []
    for i, row in enumerate(spectrum):
        if row[-1] == 1:
            conf = [j for j, a in enumerate(row[:-1]) if a == 1]
            conflicts.append(conf)

    # compute diagnoses
    diagnoses = functions.conflict_directed_search(conflicts)

    # sort diagnoses
    for d in diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(diagnoses)
    return diagnoses_sorted

def diagnosis_1(local_spectra):
    """
    go over the agents, each agent computes the diagnoses it can
    with the information it has and then passes it to the next agent
    :param local_spectra: the local spectra of the agents
    :return: a set of diagnoses
    """
    diagnoses = []
    a = 0
    while a < len(local_spectra):
        print(f'agent {a} computes diagnoses given previous diagnosis set {diagnoses}')
        # diagnoses = compute_diagnoses(local_spectra[a], diagnoses)
        local_spectrum = local_spectra[a]
        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, a in enumerate(row[:-1]) if a == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        print(f'agent {a} sends to agent {a+1} the diagnoses {diagnoses}')
        a += 1

    # sort diagnoses
    for d in diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(diagnoses)
    return diagnoses_sorted
