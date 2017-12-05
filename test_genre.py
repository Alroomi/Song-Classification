import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_genre import gen_config
from models.ann import Classifier 
from models.cnn import CNNClassifier
from feature_extractor import FeatureExtractor
from sklearn.utils.extmath import softmax
from utils import parse_csv_list_file

def load_data(fname):
    with open(fname, 'rb') as fin:
        return pkl.load(fin)

def eval(preds, labels):
    n_corrects = np.sum(preds == labels)
    return float(n_corrects) / preds.shape[0] 

def test(data_dir, list_fname, ft, model, cfgs, id_to_cls):
    all_preds = list([])
    all_labels = list([])

    mean = None
    if cfgs['zero_mean']:
        mean = np.load('means/mean_genre.npy')

    with open(list_fname, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            fname, label = line.strip().split('::')
            label = int(label.strip())
            if not os.path.exists(os.path.join(data_dir, fname)):
                continue
            feats = ft.extract_feature(os.path.join(data_dir, fname))
            if cfgs['zero_mean']:
                feats -= mean
            preds, scores = model.predict(feats) 
            if cfgs['ensemble_mode'] == 'voting':
                counts = np.bincount(preds)
                final_pred = np.argmax(counts)
            elif cfgs['ensemble_mode'] == 'score_sum':
                sum_scores = scores.sum(axis=0)
                final_pred = np.argmax(sum_scores)
            verdict = 'CORRECT' if final_pred == label else 'INCORRECT'
            print('------------')
            print(fname)
            print('Prediction: %s ==> %s' %(id_to_cls[final_pred], verdict))
            all_preds.append(final_pred)
            all_labels.append(label)

    acc = eval(np.array(all_preds), np.array(all_labels))
    print('************************************')
    print('Accuracy: %.02f %%' %(acc*100))
    print('************************************')


def test_from_feat(data_dir, list_fname, model, task, cfgs):
    all_preds = list([])
    all_labels = list([])
    samples, cls_to_id, id_to_cls = parse_csv_list_file(list_fname)
    
    mean = None
    if cfgs['zero_mean']:
        mean = np.load('means/mean_%s_%s.npy' %(task,cfgs['model']))
    n_labels = len(id_to_cls.keys())
    confusion_mat = np.zeros((n_labels, n_labels))
    
    res_fout = open('results/result_%s_%s.csv' %(task, cfgs['model']), 'w')
    res_fout.write('filename:predicted:correct?\n')
    all_scores = list([])
    for sample in samples:
        fname, label = sample 
        if not os.path.exists(os.path.join(data_dir, fname.replace('.mp3', '.npy'))):
            continue
        feats = np.load(os.path.join(data_dir, fname.replace('.mp3', '.npy')))
        if cfgs['zero_mean']:
            feats -= mean
        preds, scores = model.predict(feats)
        if cfgs['ensemble_mode'] == 'voting':
            counts = np.bincount(preds)
            final_pred = np.argmax(counts)
        elif cfgs['ensemble_mode'] == 'score_sum':
            sum_scores = scores.sum(axis=0)
            final_pred = np.argmax(sum_scores)
            sum_scores = sum_scores.reshape([1,-1])
            all_scores.append(softmax(sum_scores))
        verdict = 'CORRECT' if final_pred == label else 'INCORRECT'
        print('------------')
        print(fname)
        print('Prediction: %s ==> %s' %(id_to_cls[final_pred], verdict))
        res_fout.write('%s:%s:%s\n' %(fname, id_to_cls[final_pred], verdict.lower()))
        confusion_mat[final_pred, label] += 1
        all_preds.append(final_pred)
        all_labels.append(label)
    res_fout.close()

    acc = eval(np.array(all_preds), np.array(all_labels))
    print('************************************')
    print('Accuracy: %.02f %%' %(acc*100))
    print('************************************')
    print(confusion_mat)
    all_scores = np.vstack(all_scores)
    np.save('results/scores_%s_%s' %(task,cfgs['model']), all_scores)

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Invalid arguments'
    task = sys.argv[1].strip()
    model = sys.argv[2].strip()
    cfgs, nn_cfgs = gen_config(task, model)

    # load model 
    if nn_cfgs['model'] == 'ann':
        model = Classifier(nn_cfgs, log_dir=None)
        feat_folder = cfgs['file_all_feat_folder']
    elif nn_cfgs['model'] == 'cnn': 
        model = CNNClassifier(nn_cfgs, log_dir=None)
        feat_folder = cfgs['file_feat_folder']
    else:
        assert False, 'Invalid model type'
    model.restore('%s-%d' %(nn_cfgs['tf_sess_path'], cfgs['n_iters']))

    test_from_feat(feat_folder, cfgs['test_list_fname'], model, cfgs['task'], nn_cfgs)

