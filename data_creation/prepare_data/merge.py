import numpy as np
# s0 = np.load('Data/CP_data/GMP_1960_train0.npy')
# a0 = np.load('Data/CP_data/GMP_1960_train0_ans.npy')
# s1 = np.load('Data/CP_data/GMP_1960_train1.npy')
# a1 = np.load('Data/CP_data/GMP_1960_train1_ans.npy')
# s2 = np.load('Data/CP_data/GMP_1960_train2.npy')
# a2 = np.load('Data/CP_data/GMP_1960_train2_ans.npy')
# s3 = np.load('Data/CP_data/GMP_1960_train3.npy')
# a3 = np.load('Data/CP_data/GMP_1960_train3_ans.npy')
# s4 = np.load('Data/CP_data/GMP_1960_train4.npy')
# a4 = np.load('Data/CP_data/GMP_1960_train4_ans.npy')
# s5 = np.load('Data/CP_data/GMP_1960_train5.npy')
# a5 = np.load('Data/CP_data/GMP_1960_train5_ans.npy')
# s6 = np.load('Data/CP_data/GMP_1960_train6.npy')
# a6 = np.load('Data/CP_data/GMP_1960_train6_ans.npy')
# s7 = np.load('Data/CP_data/GMP_1960_train7.npy')
# a7 = np.load('Data/CP_data/GMP_1960_train7_ans.npy')
# s8 = np.load('Data/CP_data/GMP_1960_train8.npy')
# a8 = np.load('Data/CP_data/GMP_1960_train8_ans.npy')
# s9 = np.load('Data/CP_data/GMP_1960_train9.npy')
# a9 = np.load('Data/CP_data/GMP_1960_train9_ans.npy')
# s = np.concatenate((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9))
# print(s.shape)
# a = np.concatenate((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_1960_train.npy', s)
# np.save('Data/CP_data/GMP_1960_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_1960_valid.npy')
# av = np.load('Data/CP_data/GMP_1960_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_1960_valid.npy', sv)
# np.save('Data/CP_data/GMP_1960_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_1960_test.npy')
# at = np.load('Data/CP_data/GMP_1960_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_1960_test.npy', st)
# np.save('Data/CP_data/GMP_1960_test_ans.npy', at)

# [49433, 13704, 6504, 9053, 4734, 12992, 11654, 20728, 17688, 9819, 29]
# (156338, 512, 4)

# s0 = np.load('Data/CP_data/GMP_1750_train0.npy')
# a0 = np.load('Data/CP_data/GMP_1750_train0_ans.npy')
# s = s0
# print(s.shape)
# a = a0
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_1750_train.npy', s)
# np.save('Data/CP_data/GMP_1750_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_1750_valid.npy')
# av = np.load('Data/CP_data/GMP_1750_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_1750_valid.npy', sv)
# np.save('Data/CP_data/GMP_1750_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_1750_test.npy')
# at = np.load('Data/CP_data/GMP_1750_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_1750_test.npy', st)
# np.save('Data/CP_data/GMP_1750_test_ans.npy', at)

# [695, 169, 150, 284, 34, 851, 1119, 501, 2104, 3221, 0]
# (9128, 512, 4)

# up 8, 8

# s0 = np.load('Data/CP_data/GMP_1820_train0.npy')
# a0 = np.load('Data/CP_data/GMP_1820_train0_ans.npy')
# s1 = np.load('Data/CP_data/GMP_1820_train1.npy')
# a1 = np.load('Data/CP_data/GMP_1820_train1_ans.npy')
# s = np.concatenate((s0, s1))
# print(s.shape)
# a = np.concatenate((a0, a1))
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_1820_train.npy', s)
# np.save('Data/CP_data/GMP_1820_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_1820_valid.npy')
# av = np.load('Data/CP_data/GMP_1820_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_1820_valid.npy', sv)
# np.save('Data/CP_data/GMP_1820_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_1820_test.npy')
# at = np.load('Data/CP_data/GMP_1820_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_1820_test.npy', st)
# np.save('Data/CP_data/GMP_1820_test_ans.npy', at)

# [4668, 1582, 568, 1172, 1333, 4241, 3663, 7674, 4644, 5448, 26]
#  (35019, 512, 4)

# up 8, 8
# [18262, 1582, 568, 1172, 1333, 4241, 3663, 7674, 4644, 5448, 26]
# (48637, 512, 4)

# s0 = np.load('Data/CP_data/GMP_1860_train0.npy')
# a0 = np.load('Data/CP_data/GMP_1860_train0_ans.npy')
# s1 = np.load('Data/CP_data/GMP_1860_train1.npy')
# a1 = np.load('Data/CP_data/GMP_1860_train1_ans.npy')
# s2 = np.load('Data/CP_data/GMP_1860_train2.npy')
# a2 = np.load('Data/CP_data/GMP_1860_train2_ans.npy')
# s3 = np.load('Data/CP_data/GMP_1860_train3.npy')
# a3 = np.load('Data/CP_data/GMP_1860_train3_ans.npy')
# s = np.concatenate((s0, s1, s2, s3))
# print(s.shape)
# a = np.concatenate((a0, a1, a2, a3))
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_1860_train.npy', s)
# np.save('Data/CP_data/GMP_1860_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_1860_valid.npy')
# av = np.load('Data/CP_data/GMP_1860_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_1860_valid.npy', sv)
# np.save('Data/CP_data/GMP_1860_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_1860_test.npy')
# at = np.load('Data/CP_data/GMP_1860_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_1860_test.npy', st)
# np.save('Data/CP_data/GMP_1860_test_ans.npy', at)

# [16023, 6851, 4341, 5748, 2662, 8722, 7075, 17769, 10847, 8865, 42]
# (88945, 512, 4)

# up 2, 4
# [16023, 3446, 2179, 2887, 1339, 4382, 3561, 8915, 5450, 4456, 22]
# (14318, 512, 4)

# up 4, 8
# [31607, 6851, 4341, 5748, 2662, 8722, 7075, 17769, 10847, 8865, 42]
# (104529, 512, 4)

# s0 = np.load('Data/CP_data/GMP_1910_train0.npy')
# a0 = np.load('Data/CP_data/GMP_1910_train0_ans.npy')
# s1 = np.load('Data/CP_data/GMP_1910_train1.npy')
# a1 = np.load('Data/CP_data/GMP_1910_train1_ans.npy')
# s2 = np.load('Data/CP_data/GMP_1910_train2.npy')
# a2 = np.load('Data/CP_data/GMP_1910_train2_ans.npy')
# s3 = np.load('Data/CP_data/GMP_1910_train3.npy')
# a3 = np.load('Data/CP_data/GMP_1910_train3_ans.npy')
# s4 = np.load('Data/CP_data/GMP_1910_train4.npy')
# a4 = np.load('Data/CP_data/GMP_1910_train4_ans.npy')
# s5 = np.load('Data/CP_data/GMP_1910_train5.npy')
# a5 = np.load('Data/CP_data/GMP_1910_train5_ans.npy')
# s6 = np.load('Data/CP_data/GMP_1910_train6.npy')
# a6 = np.load('Data/CP_data/GMP_1910_train6_ans.npy')
# s7 = np.load('Data/CP_data/GMP_1910_train7.npy')
# a7 = np.load('Data/CP_data/GMP_1910_train7_ans.npy')
# s8 = np.load('Data/CP_data/GMP_1910_train8.npy')
# a8 = np.load('Data/CP_data/GMP_1910_train8_ans.npy')
# s9 = np.load('Data/CP_data/GMP_1910_train9.npy')
# a9 = np.load('Data/CP_data/GMP_1910_train9_ans.npy')
# s = np.concatenate((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9))
# print(s.shape)
# a = np.concatenate((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_1910_train.npy', s)
# np.save('Data/CP_data/GMP_1910_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_1910_valid.npy')
# av = np.load('Data/CP_data/GMP_1910_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_1910_valid.npy', sv)
# np.save('Data/CP_data/GMP_1910_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_1910_test.npy')
# at = np.load('Data/CP_data/GMP_1910_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_1910_test.npy', st)
# np.save('Data/CP_data/GMP_1910_test_ans.npy', at)

# [46616, 12891, 7955, 9337, 4490, 15384, 11998, 23888, 20937, 12649, 34]
# (166179, 512, 4)

# s0 = np.load('Data/CP_data/GMP_2023_train0.npy')
# a0 = np.load('Data/CP_data/GMP_2023_train0_ans.npy')
# s1 = np.load('Data/CP_data/GMP_2023_train1.npy')
# a1 = np.load('Data/CP_data/GMP_2023_train1_ans.npy')
# s2 = np.load('Data/CP_data/GMP_2023_train2.npy')
# a2 = np.load('Data/CP_data/GMP_2023_train2_ans.npy')
# s3 = np.load('Data/CP_data/GMP_2023_train3.npy')
# a3 = np.load('Data/CP_data/GMP_2023_train3_ans.npy')
# s4 = np.load('Data/CP_data/GMP_2023_train4.npy')
# a4 = np.load('Data/CP_data/GMP_2023_train4_ans.npy')
# s5 = np.load('Data/CP_data/GMP_2023_train5.npy')
# a5 = np.load('Data/CP_data/GMP_2023_train5_ans.npy')
# s6 = np.load('Data/CP_data/GMP_2023_train6.npy')
# a6 = np.load('Data/CP_data/GMP_2023_train6_ans.npy')
# s7 = np.load('Data/CP_data/GMP_2023_train7.npy')
# a7 = np.load('Data/CP_data/GMP_2023_train7_ans.npy')
# s8 = np.load('Data/CP_data/GMP_2023_train8.npy')
# a8 = np.load('Data/CP_data/GMP_2023_train8_ans.npy')
# s9 = np.load('Data/CP_data/GMP_2023_train9.npy')
# a9 = np.load('Data/CP_data/GMP_2023_train9_ans.npy')
# s10 = np.load('Data/CP_data/GMP_2023_train10.npy')
# a10 = np.load('Data/CP_data/GMP_2023_train10_ans.npy')
# s11 = np.load('Data/CP_data/GMP_2023_train11.npy')
# a11 = np.load('Data/CP_data/GMP_2023_train11_ans.npy')
# s = np.concatenate((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11))
# print(s.shape)
# a = np.concatenate((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
# a = np.around(np.emath.logn(1.5157, a+1), 2)
# print(a.shape)
# print(np.unique(a))
# hist, _ = np.histogram(a, bins=range(12))
# print(hist)
# np.save('Data/CP_data/GMP_2023_train.npy', s)
# np.save('Data/CP_data/GMP_2023_train_ans.npy', a)

# sv = np.load('Data/CP_data/GMP_2023_valid.npy')
# av = np.load('Data/CP_data/GMP_2023_valid_ans.npy')
# av = np.around(np.emath.logn(1.5157, av+1), 2)
# np.save('Data/CP_data/GMP_2023_valid.npy', sv)
# np.save('Data/CP_data/GMP_2023_valid_ans.npy', av)

# st = np.load('Data/CP_data/GMP_2023_test.npy')
# at = np.load('Data/CP_data/GMP_2023_test_ans.npy')
# at = np.around(np.emath.logn(1.5157, at+1), 2)
# np.save('Data/CP_data/GMP_2023_test.npy', st)
# np.save('Data/CP_data/GMP_2023_test_ans.npy', at)

# [71502, 16485, 8868, 7998, 4779, 18973, 16691, 24374, 20928, 15413, 118]
# (206129, 512, 4)