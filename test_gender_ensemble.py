import os, os.path, sys
import numpy as np
import pickle as pkl
from configs_gender import gen_config
from utils import parse_csv_list_file

def load_data(fname):
    with open(fname, 'rb') as fin:
        return pkl.load(fin)

def eval(preds, labels):
    n_corrects = np.sum(preds == labels)
    return float(n_corrects) / preds.shape[0] 

def test_ensemble(data_dir, list_fname, task, cfgs):
    all_preds = list([])
    all_labels = list([])
    samples, cls_to_id, id_to_cls = parse_csv_list_file(list_fname)
    
    mean = None
    n_labels = len(id_to_cls.keys())
    confusion_mat = np.zeros((n_labels, n_labels))

    res_fout = open('results/result_%s_ensemble.csv' %task, 'w')
    res_fout.write('filename:predicted:correct?\n')
    cnn_scores = np.load('results/scores_%s_cnn.npy' %task)
    ann_scores = np.load('results/scores_%s_ann.npy' %task)
    final_scores = cnn_scores + ann_scores
    counter = 0
    for sample in samples:
        fname, label = sample 
        if not os.path.exists(os.path.join(data_dir, fname.replace('.mp3', '.npy'))):
            counter += 1
            continue
        final_pred = np.argmax(final_scores[counter,:])
        verdict = 'CORRECT' if final_pred == label else 'INCORRECT'
        print('------------')
        print(fname)
        print('Prediction: %s ==> %s' %(id_to_cls[final_pred], verdict))
        res_fout.write('%s:%s:%s\n' %(fname, id_to_cls[final_pred], verdict.lower()))
        confusion_mat[final_pred, label] += 1
        all_preds.append(final_pred)
        all_labels.append(label)
        counter += 1
    res_fout.close()

    acc = eval(np.array(all_preds), np.array(all_labels))
    print('************************************')
    print('Accuracy: %.02f %%' %(acc*100))
    print('************************************')
    print(confusion_mat)

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Invalid arguments'
    task = sys.argv[1].strip() 
    model = sys.argv[2].strip()
    cfgs, nn_cfgs = gen_config(task, model)

    if nn_cfgs['model'] == 'ann':
        feat_folder = cfgs['file_all_feat_folder']
    elif nn_cfgs['model'] == 'cnn': 
        feat_folder = cfgs['file_feat_folder']
    else:
        assert False, 'Invalid model type'
    test_ensemble(feat_folder, cfgs['test_list_fname'], cfgs['task'], nn_cfgs)
