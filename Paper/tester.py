from Paper import methods_for_diagnosis, methods_for_input_preprocess

noa = 7

S = [
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 1, 0, 0]
]

# prepare inputs: divide the spectrum to local spectra - one for each agent
local_spectra, missing_row_numbers = methods_for_input_preprocess.input_preprocess_1(noa, S)

# run the algorithm, collect diagnosis messages count
diagnoses, info_sent_diagnosis, revealed_information_sum, revealed_information_mean,\
        revealed_information_mean_per_agent, revealed_information_last = methods_for_diagnosis.diagnosis_1(local_spectra)

print(9)
