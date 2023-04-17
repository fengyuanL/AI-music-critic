import numpy as np
ans = np.load("F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_valid_ans.npy")
ans = np.around(np.emath.logn(1.5157, ans+1), 2)
np.save("F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_valid_ans.npy", ans)
ans = np.load("F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_test_ans.npy")
ans = np.around(np.emath.logn(1.5157, ans+1), 2)
np.save("F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_test_ans.npy", ans)