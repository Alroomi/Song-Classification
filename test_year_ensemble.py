import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_year import gen_config
from models.ann import Classifier 
from models.cnn import CNNClassifier 
from feature_extractor import FeatureExtractor
from train_year import build_label_map_year2decade
from utils import parse_csv_list_file

def load_pkl(fname):
    with open(fname, 'rb') as fin:
        data = pkl.load(fin)
    return data 

def eval(preds, labels):
    n_corrects = np.sum(preds == labels)
    return float(n_corrects) / preds.shape[0] 

def eval_delta(preds, labels, delta=0):
    n_corrects = np.sum(np.abs(preds - labels) <= delta)
    return float(n_corrects) / preds.shape[0] 

def test_custom_label_from_feat(data_dir, list_fname, task, cfgs):
    all_preds = list([])
    all_labels = list([])
    samples, cls_to_id, _ = parse_csv_list_file(list_fname)
    old_to_new_ids, new_id_to_cls, _ = build_label_map_year2decade(cls_to_id)

    mean = None
    n_labels = len(new_id_to_cls.keys())
    confusion_mat = np.zeros((n_labels, n_labels))

    res_fout = open('results/result_%s_decade_ensemble.csv' %task, 'w')
    res_fout.write('filename:predicted:correct?\n')
    cnn_scores = np.load('results/scores_%s_decade_cnn.npy' %task)
    ann_scores = np.load('results/scores_%s_decade_ann.npy' %task)
    final_scores = cnn_scores + ann_scores    
    
    with open(list_fname, 'r') as fin:
        lines = fin.readlines()
        counter = 0

        for sample in samples:
            fname, label = sample 
            if not os.path.exists(os.path.join(data_dir, fname.replace('.mp3', '.npy'))):
                counter += 1
                continue
            final_pred = np.argmax(final_scores[counter,:])
            verdict = 'CORRECT' if final_pred == old_to_new_ids[label] else 'INCORRECT'
            print('------------')
            print('%s -- (label: %s)' %(fname,new_id_to_cls[old_to_new_ids[label]]))
            print('Prediction: %s ==> %s' %(new_id_to_cls[final_pred], verdict))
            res_fout.write('%s:%s:%s\n' %(fname, new_id_to_cls[final_pred], verdict.lower()))
            confusion_mat[final_pred, old_to_new_ids[label]] += 1
            all_preds.append(final_pred)
            all_labels.append(old_to_new_ids[label])
            counter += 1

    res_fout.close()
    acc = eval(np.array(all_preds), np.array(all_labels))
    print('************************************')
    print('Accuracy: %.02f %%' %(acc*100))
    print('************************************')
    print(confusion_mat)

def test_year_from_feat(data_dir, list_fname, task, cfgs, delta=3):
    all_preds = list([])
    all_labels = list([])
    samples, cls_to_id, id_to_cls = parse_csv_list_file(list_fname)

    mean = None

    res_fout = open('results/result_%s_year_ensemble.csv' %task, 'w')
    res_fout.write('filename:predicted:correct?\n')
    cnn_scores = np.load('results/scores_%s_year_cnn.npy' %task)
    ann_scores = np.load('results/scores_%s_year_ann.npy' %task)
    final_scores = cnn_scores + ann_scores    
    
    with open(list_fname, 'r') as fin:
        lines = fin.readlines()
        counter = 0

        for sample in samples:
            fname, label = sample 
            if not os.path.exists(os.path.join(data_dir, fname.replace('.mp3', '.npy'))):
                counter += 1
                continue
            final_pred = np.argmax(final_scores[counter,:])
            verdict = 'CORRECT' if abs(final_pred - label) <= delta else 'INCORRECT'
            print('------------')
            print('%s -- (label: %s)' %(fname,label))
            print('Prediction: %s ==> %s' %(final_pred, verdict))
            res_fout.write('%s:%d:%s\n' %(fname, final_pred, verdict.lower()))
            all_preds.append(final_pred)
            all_labels.append(label)
            counter += 1

    res_fout.close()
    acc = eval_delta(np.array(all_preds), np.array(all_labels), delta)
    print('************************************')
    print('Accuracy: %.02f %%' %(acc*100))
    print('************************************')

if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Invalid arguments'
    task = sys.argv[1].strip() 
    model = sys.argv[2].strip()
    custom_label = sys.argv[3].strip()
    cfgs, nn_cfgs = gen_config(task, model, custom_label)

    if nn_cfgs['model'] == 'ann':
        feat_folder = cfgs['file_all_feat_folder']
    elif nn_cfgs['model'] == 'cnn': 
        feat_folder = cfgs['file_feat_folder']
    else:
        assert False, 'Invalid model type'

    if custom_label == 'decade':
        test_custom_label_from_feat(feat_folder, cfgs['test_list_fname'], cfgs['task'], nn_cfgs)
    elif custom_label == 'year':
        test_year_from_feat(feat_folder, cfgs['test_list_fname'], cfgs['task'], nn_cfgs, delta=3)

