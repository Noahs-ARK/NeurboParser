# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 20:44:45 2016

@author: hpeng
"""
import io
UNK = '__UNK__'
def build_vocab(base_dir = '../semeval2015_data/dm/data/english'):
    files = ['english_dm_augmented_train.sdp',
             'english_dm_augmented_dev.sdp',
             'english_id_dm_augmented_test.sdp', 
             'english_ood_dm_augmented_test.sdp'
             ]
    word2idx = {}; n = 0
    for f in files:
        path = '%s/%s' % (base_dir, f)
        with io.open(path, 'r', encoding = 'utf_8') as fin:
            for line in fin:
                ws = line.strip().split('\t')
                if len(ws) <= 1:
                    continue
                word = ws[1].lower()
                if word not in word2idx:
                    word2idx[word] = n
                    n += 1
    return word2idx

def prune_embedding(infile = '.', word2idx = None):
    path = '%s/glove.6B.100d.txt' % infile
    fout = io.open('%s/glove.100.pruned' % infile, 'w', encoding = 'utf-8')
    n = 0
    with io.open(path, 'r', encoding = 'utf-8') as fin:
        for line in fin:
            ws = line.strip().split()
            if len(ws) <= 5:
                continue
            word = ws[0]
            if word.lower() not in word2idx:
                continue
            
            ws[0] = ws[0].lower()
            o_line = ' '.join(ws)
            fout.write(o_line + u'\n')
            n += 1
    print 'Number of embeddings in vocabulary: %d' % n
    fout.close()
              
if __name__ == '__main__':
    base_dir = '../semeval2015_data/dm/data/english'
    word2idx = build_vocab(base_dir)
    prune_embedding(infile = '.', word2idx = word2idx)