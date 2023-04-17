import torch
a = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
b = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])


def accuracy(t1, t2):
    if len(t1) != len(t2):
        print("Warning: Not equal length in accuracy")
    if len(t1) > len(t2):
        t1 = t1[:len(t2)]
    elif len(t2) > len(t1):
        t2 = t2[:len(t1)]
    return torch.tensor(sum(torch.logical_and(t1, t2)) / len(t1), dtype=float)

y = a
output = b
threshold1 = 2
threshold2 = 5

acc0 = accuracy(y <= threshold1, output <= threshold1)
acc1 = accuracy(torch.logical_and(y > threshold1, y <= threshold2),
                     torch.logical_and(output > threshold1, output <= threshold2))
acc2 = accuracy(y > threshold2, output > threshold2)
acc = acc0 + acc1 + acc2
print(acc, acc0, acc1, acc2)