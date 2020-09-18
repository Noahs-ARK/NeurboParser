# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import io, numpy as np

def write_file(base_dir='.', split='train'):
    path = '%s/fn1.5.%s.conll.bios' % (base_dir, split)
    lines = io.open(path, 'r', encoding='utf-8').readlines()
    fout = io.open(split, 'w', encoding='utf-8')
    out_lines = []
    prev_id = '__unk__'
    prev_bio = '__unk__'
    for line in lines:
        ts = line.strip().split('\t')
        if len(ts) >= 2:
            curr_id = ts[6]
            if prev_id != curr_id:
                if prev_id != '__unk__':
                    out_lines.insert(len(out_lines) - 1, u"# end_of_instance\n")
                out_lines.append("# instance_id: %s" % curr_id + u"\n")
                prev_id = curr_id
            curr_bio = ts[-1]
            if curr_bio != '_' and prev_bio == curr_bio and curr_bio[0] != 'I':
                assert curr_bio[0] == 'S'
                ts[-1] = u'O-%s' % (curr_bio.split('-'))[1]
                prev_ts = out_lines[-1].strip().split('\t')
                prev_ts[-1] = u'B-%s' % (prev_bio.split('-'))[1]
                out_lines[-1] = '\t'.join(prev_ts) + u'\n'
                out_lines.append('\t'.join(ts) + u'\n')
                continue
            prev_bio = curr_bio
            
        out_lines.append(line)
    out_lines.insert(len(out_lines) - 1, u"# end_of_instance\n")
    for line in out_lines:
        fout.write(line)
    fout.close()
    
def f():
    path = 'fn1.5.train.conll.bios'
    lines = io.open(path, 'r', encoding='utf-8').readlines()
    
    role_dict_train, fram_dict_train = {}, {}
    posbio_dict_train, roleposbio_dict_train, pos_dict_train = {}, {}, {}
    s_idx, e_idx, span_len = -1, -1, -1
    for line in lines:
        ts = line.strip().split('\t')
        if len(ts) <= 2:
            continue
        bio_role = ts[-1]
        pos = ts[5]
        pos_dict_train[pos] = pos_dict_train.get(pos, 0) + 1
        if bio_role != '_':
            bio, role = bio_role.split('-')
            if bio == 'S':
                role_dict_train[role] = max(role_dict_train.get(role, 0), 1)
                posbio_dict_train[(pos, 'S')] = posbio_dict_train.get((pos, 'S'), 0) + 1
                roleposbio_dict_train[(role, pos, 'S')] = roleposbio_dict_train.get((role, pos, 'S'), 0) + 1
            elif bio == 'B':
                assert span_len == -1
                span_len = 1
                posbio_dict_train[(pos, 'B')] = posbio_dict_train.get((pos, 'B'), 0) + 1
                roleposbio_dict_train[(role, pos, 'B')] = role, posbio_dict_train.get((role, pos, 'B'), 0) + 1
            elif bio == 'I':
                assert span_len >= 1
                span_len += 1
                posbio_dict_train[(pos, 'I')] = posbio_dict_train.get((pos, 'I'), 0) + 1
                roleposbio_dict_train[(role, pos, 'I')] = role, posbio_dict_train.get((role, pos, 'I'), 0) + 1
            elif bio == 'O':
                span_len += 1
                role_dict_train[role] = max(role_dict_train.get(role, 0), span_len)
                span_len = -1
                posbio_dict_train[(pos, 'O')] = posbio_dict_train.get((pos, 'O'), 0) + 1
                roleposbio_dict_train[(role, pos, 'O')] = role, posbio_dict_train.get((role, pos, 'O'), 0) + 1
                
    print posbio_dict_train
    for pos in pos_dict_train:
        for bio in ['S', 'B', 'I', 'O']:
            print pos, bio, posbio_dict_train.get((pos, bio), 0)
    path = 'fn1.5.test.conll.bios'
    lines = io.open(path, 'r', encoding='utf-8').readlines()
    missed, total = 0, 0
    max_len, span_len = 100, -1
    pos_threshold = 1
    total_len = 0
    for line in lines:
        ts = line.strip().split('\t')
        if len(ts) <= 2:
            continue
        bio_role = ts[-1]
        pos = ts[5]
        if bio_role != '_':
            bio, role = bio_role.split('-')
            if bio == 'S':
                span_len = 1
                total += 1
                if roleposbio_dict_train.get((role, pos, bio), 0) < pos_threshold:
                    missed += 1
                #print role, role_dict_train.get(role, np.inf)
                #total_len += min(max_len, role_dict_train.get(role, np.inf))
                #if span_len > min(max_len, role_dict_train.get(role, np.inf)):
                #    missed += 1
                span_len = -1
            elif bio == 'B':
                if roleposbio_dict_train.get((role, pos, bio), 0) < pos_threshold:
                    missed += 1
                assert span_len == -1
                span_len = 1
            elif bio == 'I':
                if roleposbio_dict_train.get((role, pos, bio), 0) < pos_threshold:
                    missed += 1
                assert span_len >= 1
                span_len += 1
            elif bio == 'O':
                span_len += 1
                total += 1
                if roleposbio_dict_train.get((role, pos, bio), 0) < pos_threshold:
                    missed += 1
                #print role, role_dict_train.get(role, np.inf)
                #total_len += min(max_len, role_dict_train.get(role, np.inf))
                #if span_len > min(max_len, role_dict_train.get(role, np.inf)):
                #    missed += 1
                span_len = -1
    #for k, v in role_dict_train.items():
    #    print k, v
    print '%d/%d = %f' % (missed, total, missed * 1.0 / total)
    #print total_len * 1.0 / total
    dsa
if __name__ == '__main__':
    f()
    dsadsds
    write_file()
    
    dsa
    count, total = 0, 0
    pos_dict, pos_dict_ = {}, {}
    for line in lines:
        ts = line.split('\t')
        if len(ts) <= 4:
            continue
        if ts[1] != ts[3]:
            print ts[1], ts[3]
            count += 1
        pos_dict[ts[4]] = pos_dict.get(ts[4], 0) + 1
        pos_dict_[ts[4]] = pos_dict_.get(ts[4], 0) + 1
        total += 1
    print count, total
    
    print len(pos_dict), len(pos_dict_)
