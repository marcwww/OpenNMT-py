
atec_train = 'atec_train.txt'
atec_valid = 'atec_valid.txt'
seqs1=[]
seqs2=[]
lbls=[]

with open(atec_train, 'r') as f_in, open(atec_train+'.back', 'w') as f_out:
    for line in f_in:
        seq1, seq2, lbl=line.split('\t')
        seq1 = ''.join(seq1.split(' '))
        seq2 = ''.join(seq2.split(' '))
        seqs1.append(seq1)
        seqs2.append(seq2)
        lbls.append(lbl)
        f_out.write(seq1+'\t'+seq2+'\t'+lbl)

with open(atec_valid, 'r') as f_in, open(atec_valid+'.back', 'w') as f_out:
    for line in f_in:
        seq1, seq2, lbl=line.split('\t')
        seq1 = ''.join(seq1.split(' '))
        seq2 = ''.join(seq2.split(' '))
        seqs1.append(seq1)
        seqs2.append(seq2)
        lbls.append(lbl)
        f_out.write(seq1+'\t'+seq2+'\t'+lbl)

with open(atec_train+'.'+atec_valid+'.back','w') as f_out:
    for seq1,seq2,lbl in zip(seqs1,seqs2,lbls):
        f_out.write(seq1 + '\t' + seq2 + '\t' + lbl)