from Paper import methods_for_ranking

spectrum = [
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 0]
]
local_spectra = [
    [
        [1, 1, 0, 1],
        [2, 2, 2, 2],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ],
    [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ],
    [
        [2, 2, 2, 2],
        [0, 1, 1, 1],
        [2, 2, 2, 2],
        [1, 0, 1, 0]
    ]
]

diagnoses = [[0, 1], [0, 2]]

ranked_diagnoses00 = methods_for_ranking.ranking_0(spectrum, diagnoses, 0.5)

# ranked_diagnoses01, info_sent_ranking01 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.1)
# ranked_diagnoses02, info_sent_ranking02 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.2)
# ranked_diagnoses03, info_sent_ranking03 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.3)
# ranked_diagnoses04, info_sent_ranking04 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.4)
ranked_diagnoses05, info_sent_ranking05 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.5)
# ranked_diagnoses06, info_sent_ranking06 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.6)
# ranked_diagnoses07, info_sent_ranking07 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.7)
# ranked_diagnoses08, info_sent_ranking08 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.8)
# ranked_diagnoses09, info_sent_ranking09 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 0.913)
# ranked_diagnoses10, info_sent_ranking10 = methods_for_ranking.ranking_1(local_spectra, diagnoses, 1.0)

print(9)
