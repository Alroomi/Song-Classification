import sys
import numpy as np 

def acc_from_cm(cm):
    n_labels = cm.shape[0]
    acc = float(np.sum([cm[i,i] for i in range(n_labels)])) / np.sum(cm)
    return acc

def per_label_acc(cm, ind_to_label):
    n_labels = cm.shape[0]
    label_acc = dict({})
    for i in range(n_labels):
        if np.sum(cm[i,:]) == 0:
            label_acc[ind_to_label[i]] = 0
        else:
            label_acc[ind_to_label[i]] = float(cm[i,i]) / np.sum(cm[i,:])
    return label_acc

def parse_labels(fname, label_col_ind, delimiter=';'):
    label_to_ind = dict({})
    ind_to_label = dict({})
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    n_lines = len(lines)
    headers = lines[0].strip().split(delimiter)
    n_labels = 0
    for i in range(1, n_lines):
        label = lines[i].strip().split(delimiter)[label_col_ind].strip()
        label = label.lower()
        if label not in label_to_ind:
            label_to_ind[label] = n_labels
            ind_to_label[n_labels] = label
            n_labels += 1
    label_to_ind['unknown'] = n_labels
    ind_to_label[n_labels] = 'unknown'
    n_labels += 1
    return label_to_ind, ind_to_label

def parse_result_file(fname, label_col_ind, label_to_ind, ind_to_label, delimiter=';'):
    ''' label_col_ind: index of column with groudtruth labels
    '''
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    n_lines = len(lines)
    headers = lines[0].strip().split(delimiter)
    res_dict = dict({})

    gt = dict({})
    n_labels = len(label_to_ind.keys())

    for i in range(label_col_ind+1, len(headers)):
        res_dict[headers[i]] = dict({})
        res_dict[headers[i]]['cm'] = np.zeros((n_labels, n_labels), dtype=np.int32)

    for i in range(1, n_lines):
        splt = lines[i].strip().split(delimiter)
        label = splt[label_col_ind].lower()
        song = splt[0].strip()
        for j in range(label_col_ind+1, len(splt)):
            pred = splt[j].strip().lower()
            if pred not in label_to_ind:
                # print('Strange label (%s) found in column %s' %(pred, headers[j]))
                pred = 'unknown'
            model_name = headers[j]
            res_dict[model_name][song] = pred
            gt[song] = label
            u = label_to_ind[pred] # index of predicted label
            v = label_to_ind[label] # index of groundtruth label
            res_dict[model_name]['cm'][u,v] += 1 
    for k in res_dict:
        res_dict[k]['acc'] = acc_from_cm(res_dict[k]['cm'])
        res_dict[k]['label_acc'] = per_label_acc(res_dict[k]['cm'], ind_to_label)
    return res_dict, gt

def majority_voting(fname, label_col_ind, res_dict, label_to_ind, ind_to_label, delimiter=';'):
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    headers = lines[0].strip().split(delimiter)
    n_lines = len(lines)
    n_corrects = 0
    with open(fname.replace('.csv', '_fused_1.csv'), 'w') as fout:
        fout.write('%s;Fusion\n' %lines[0].strip())
        for i in range(1, n_lines):
            splt = lines[i].strip().split(delimiter)
            song = splt[0].strip()
            votes = np.zeros(len(label_to_ind.keys()))
            for k in res_dict.keys():
                if res_dict[k][song] not in label_to_ind:
                    continue
                votes[label_to_ind[res_dict[k][song]]] += 1
            max_votes = np.max(votes)
            if np.sum(votes==max_votes) > 1: 
                # break tie using models' overall accuracies
                indices = np.nonzero(votes == max_votes)[0]
                tmp = -1e10
                pred = None
                for k in res_dict.keys():
                    if (res_dict[k][song] != 'unknown' 
                    and label_to_ind[res_dict[k][song]] in indices 
                    and res_dict[k]['acc'] > tmp):
                        tmp = res_dict[k]['acc']
                        pred = res_dict[k][song]
            else:
                pred = ind_to_label[np.argmax(votes)]
            if pred == splt[label_col_ind].strip().lower():
                n_corrects += 1
            fout.write('%s;%s\n' %(lines[i].strip(), pred))
    print('Majority Voting Accuracy: %f' %(float(n_corrects) / (n_lines - 1)))

def score_weight(fname, label_col_ind, res_dict, label_to_ind, delimiter=';'):
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    headers = lines[0].strip().split(delimiter)
    n_lines = len(lines)
    n_corrects = 0
    with open(fname.replace('.csv', '_fused_2.csv'), 'w') as fout:
        fout.write('%s;Fusion\n' %lines[0].strip())
        for i in range(1, n_lines):
            splt = lines[i].strip().split(delimiter)
            song = splt[0].strip()
            preds = list([])
            weights = list([])
            for k in res_dict.keys():
                pred = res_dict[k][song]
                if pred not in label_to_ind:
                    continue
                preds.append(pred)
                weights.append(res_dict[k]['label_acc'][pred])
                # print('%s - %f' %(pred, res_dict[k]['label_acc'][pred]))
            ind = np.argmax(np.array(weights))
            final_pred = preds[ind]
            # print('--> %s' %final_pred)
            # import pdb; pdb.set_trace()
            if final_pred == splt[label_col_ind].strip().lower():
                n_corrects += 1
            fout.write('%s;%s\n' %(lines[i].strip(), final_pred))
    print('Score Weighting Accuracy: %f' %(float(n_corrects) / (n_lines - 1)))

def calc_pfms(gt, res_dict, label_to_ind):
    pfms = dict({})
    keys = sorted(res_dict.keys())
    n_labels = len(label_to_ind.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            pfm = np.zeros((n_labels,n_labels,n_labels), dtype=np.float32)
            for song in gt:
                label = label_to_ind[gt[song]]
                pred_1 = label_to_ind[res_dict[keys[i]][song]]
                pred_2 = label_to_ind[res_dict[keys[j]][song]]
                pfm[label, pred_1, pred_2] += 1
            tmp = np.sum(pfm, axis=0, keepdims=True)
            tmp[tmp==0] = 1e20
            pfm = pfm / tmp
            pfms['%s&%s' %(keys[i],keys[j])] = pfm 
    return pfms

def fuse_pfm(fname, label_col_ind, res_dict, pfms, label_to_ind, ind_to_label, delimiter=';'):
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    headers = lines[0].strip().split(delimiter)
    n_lines = len(lines)
    n_corrects = 0
    with open(fname.replace('.csv', '_fused_3.csv'), 'w') as fout:
        fout.write('%s;Fusion\n' %lines[0].strip())
        for i in range(1, n_lines):
            splt = lines[i].strip().split(delimiter)
            song = splt[0].strip()
            max_prob = -1.0
            pred = None
            for k in pfms:
                pfm = pfms[k]
                md1, md2 = k.split('&')
                ind1 = label_to_ind[res_dict[md1][song]]
                ind2 = label_to_ind[res_dict[md2][song]]
                probs = pfm[:,ind1,ind2]
                if probs.max() > max_prob:
                    max_prob = probs.max()
                    pred = ind_to_label[np.argmax(probs)]
            if pred == splt[label_col_ind].strip().lower():
                n_corrects += 1
            fout.write('%s;%s\n' %(lines[i].strip(), pred))
    print('PFM-based Fusion Accuracy: %f' %(float(n_corrects) / (n_lines - 1)))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        task = sys.argv[1]
        mode = sys.argv[2]
    else:
        task = 'Gender_fold_1'
        mode = 'score_weight'

    print('Task: %s' %task)
    print('Fusion method: %s' %mode)

    label_col_ind = 3 # index of the label column (0-based)
    if 'Year' in task:
        label_col_ind = 4

    full_test_fname = '%s_test.csv' %task
    val_fname = '%s_val.csv' %task
    test_fname = '%s_eval.csv' %task

    label_to_ind, ind_to_label = parse_labels(full_test_fname, label_col_ind)
    val_res_dict, gt = parse_result_file(val_fname, label_col_ind, label_to_ind, ind_to_label)
    res_dict, _ = parse_result_file(test_fname, label_col_ind, label_to_ind, ind_to_label)

    for k in sorted(val_res_dict.keys()):
        print('%s: Accuracy = %f' %(k, res_dict[k]['acc']))
        # use accuracies calculated from validation set as weights for fusion
        res_dict[k]['acc'] = val_res_dict[k]['acc']
        res_dict[k]['label_acc'] = val_res_dict[k]['label_acc']

    if mode == 'vote':
        majority_voting(test_fname, label_col_ind, res_dict, label_to_ind, ind_to_label) 
    elif mode == 'score_weight':
        score_weight(test_fname, label_col_ind, res_dict, label_to_ind)
    elif mode == 'pfm':
        pfms = calc_pfms(gt, val_res_dict, label_to_ind) 
        fuse_pfm(test_fname, label_col_ind, res_dict, pfms, label_to_ind, ind_to_label)
    else:
        assert False, 'Invalid'

