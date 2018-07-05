
A2B = 'actual/AtoB.actual.ti.final'
B2A = 'actual/AtoB.actual.ti.final'

with open(A2B, 'r') as f1, open(B2A, 'r') as f2:
    for line in f1:
        print(line.split(' '))
