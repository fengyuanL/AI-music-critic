import torch
a = torch.tensor([0, 0, 18, 18, 0])
b = torch.tensor([0, 0, 0, 18, 0])

print(a < 1)
print(b < 1)
print(torch.logical_and(a < 1, b < 1))
print(torch.logical_xor(a < 1, b < 1))

def accuracy_local(t1, t2):
    # t1 is the reference
    res = 0
    t_count = 0
    for count, i in enumerate(t1):
        if t1[count]:
            t_count += 1
            if t2[count] == True:
                res += 1
    return res/t_count

print(accuracy_local(a > 1, b > 1))
