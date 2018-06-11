import jieba
import codecs
import os
import jieba.analyse

HOME=os.path.abspath('..')
data=os.path.join(HOME,'data')

def tokenize(txt,stop_words=None):
    lines = codecs.open(txt, 'r', encoding='utf8').readlines()
    if stop_words is not None:
        jieba.analyse.set_stop_words(stop_words)

    with open(txt+'.seg','w') as f:
        for line in lines:
            seq1,seq2,lbl=line.strip().split('\t')
            seq1_seg=' '.join(list(jieba.cut(seq1)))
            seq2_seg = ' '.join(list(jieba.cut(seq2)))
            f.write(seq1_seg+'\t'
                    +seq2_seg+'\t'
                    +lbl+'\n')

if __name__=='__main__':
    # tokenize(os.path.join(data,'train'),os.path.join(data,'stop_words'))
    # tokenize(os.path.join(data,'test'),os.path.join(data,'stop_words'))

    name = 'atec_nlp_sim_train.csv'

    tokenize(os.path.join(data,name),os.path.join(data,'stop_words'))



