import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_year import gen_config
from models.cnn_regressor import CNNRegressor 
from utils import parse_csv_list_file

def load_data(fname):
    with open(fname, 'rb') as fin:
        return pkl.load(fin)

def load_pkl(fname):
    with open(fname, 'rb') as fin:
        data = pkl.load(fin)
    return data 

def test_from_feat(data_dir, list_fname, model, task, cfgs):
    all_preds = list([])
    all_labels = list([])
    samples, cls_to_id, id_to_cls = parse_csv_list_file(list_fname)

    mean = None
    if cfgs['zero_mean']:
        mean = np.load('means/mean_%s_%s.npy' %(task, cfgs['model']))

    for sample in samples:
        fname, label = sample 
        if not os.path.exists(os.path.join(data_dir, fname.replace('.mp3', '.npy'))):
            continue
        feats = np.load(os.path.join(data_dir, fname.replace('.mp3', '.npy')))
        if cfgs['zero_mean']:
            feats -= mean
        preds = model.predict(feats)
        final_pred = np.median(preds)
        final_pred = np.round(final_pred)
        print('------------')
        print('%s -- (label: %s)' %(fname,label))
        print('Prediction: %s ==> Error: %f' %(final_pred, abs(final_pred - label)))
        all_preds.append(final_pred)
        all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds += all_labels.min()
    mse = np.mean(np.square(all_preds - all_labels))
    mae = np.mean(np.abs(all_preds - all_labels))
    median_ae = np.median(np.abs(all_preds - all_labels))
    print('************************************')
    print('MSE: %.02f' %(mse))
    print('MAE: %.02f' %(mae))
    print('Median AE: %.02f' %(median_ae))
    print('************************************')

if __name__ == '__main__':
    task = sys.argv[1].strip() 
    model = 'cnn'
    custom_label = 'regressor'
    cfgs, nn_cfgs = gen_config(task, model, custom_label)

    # load model 
    model = CNNRegressor(nn_cfgs, log_dir=None)
    model.restore('%s-%d' %(nn_cfgs['tf_sess_path'], cfgs['n_iters']))

    feat_folder = cfgs['file_feat_folder']
    test_from_feat(feat_folder, cfgs['test_list_fname'], model, cfgs['task'], nn_cfgs)

    
