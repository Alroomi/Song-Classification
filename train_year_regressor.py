import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_year import gen_config
from models.cnn_regressor import CNNRegressor 
from utils import parse_csv_list_file

def load_h5py(fname):
    with h5py.File(fname, 'r') as f:
        data = f.get('feature')
        X = np.array(data)
        y = np.array(f.get('label'))
    return X, y

def load_feat(data_dir, list_fname):
    X = list([])
    y = list([])
    samples, cls_to_id, id_to_cls = parse_csv_list_file(list_fname)
    for sample in samples:
        fname, label = sample
        print(fname)
        feat_fname = fname.replace('.mp3','.npy')
        if not os.path.exists(os.path.join(data_dir, feat_fname)):
            print('File not found...Skip!')
            continue
        feats = np.load(os.path.join(data_dir, feat_fname))
        labels = np.ones(feats.shape[0], dtype=np.int8) * int(label)
        X.append(feats)
        y.append(labels)
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y, cls_to_id, id_to_cls
    
def shuffle(X, y):
    n = X.shape[0]
    rand_ind = np.random.permutation(n)
    X = X[rand_ind, ...]
    y = y[rand_ind]
    return X, y 

def split_train_val(X, y, train_ratio=0.9):
    n = X.shape[0]
    cutoff = int(n * train_ratio)
    X_tr = X[:cutoff, ...]
    y_tr = y[:cutoff, ...]
    X_val = X[cutoff:, ...]
    y_val = y[cutoff:, ...]
    return X_tr, y_tr, X_val, y_val

def next_batch(X, y, current, bs):
    start = current
    end = current + bs 
    current = end
    if end >= X.shape[0]:
        end = X.shape[0]
        current = 0
    X_batch = X[start:end,...]
    y_batch = y[start:end]
    return X_batch, y_batch, current

def _to_one_hot(y, num_classes):
    n = y.shape[0]
    y_ = np.zeros((n, num_classes))
    y = y.astype(np.int8)
    ind1 = np.arange(n)
    y_[ind1,y] = 1
    return y_

def val(model, X_val, y_val, bs):
    current = 0
    counter = 0
    val_loss = 0.0
    while True:
        X_batch, y_batch, current = next_batch(X_tr, y_tr, current, bs)
        loss = model.calc_loss(X_batch, y_batch)
        val_loss += loss 
        counter += 1
        if current == 0:
            break
    val_loss /= counter
    return val_loss

def load_pkl(fname):
    with open(fname, 'rb') as fin:
        data = pkl.load(fin)
    return data 

def save_data(fname, data):
    with open(fname, 'wb') as fout:
        pkl.dump(data, fout, pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    task = sys.argv[1].strip()
    model = 'cnn'
    custom_label = 'regressor'
    cfgs, nn_cfgs = gen_config(task, model, custom_label)
    X, y, cls_to_id, id_to_cls = load_feat(cfgs['file_feat_folder'], cfgs['train_list_fname'])

    if nn_cfgs['zero_mean']:
        mean = X.mean(axis=0, keepdims=True)
        np.save('means/mean_%s_%s' %(cfgs['task'], nn_cfgs['model']), mean)
        X -= mean

    y = y.astype(np.float32)
    y = y - y.min()
    # y_mean = y.mean()
    # y_max = y.max()
    # y = y - y_mean / (y_max - y_mean)
    y = y.reshape([-1,1])
    print(np.unique(y))

    print('Finish loading of %d samples' %X.shape[0])
    X, y = shuffle(X,y)
    
    model = CNNRegressor(nn_cfgs, nn_cfgs['log_dir'])

    X_tr, y_tr, X_val, y_val = split_train_val(X, y, train_ratio=0.9)
    lr = nn_cfgs['initial_lr']

    if not os.path.isdir(nn_cfgs['log_dir']):
        os.makedirs(nn_cfgs['log_dir'])

    bs = nn_cfgs['bs']
    current = 0
    for i in range(cfgs['n_iters']):
        X_batch, y_batch, current = next_batch(X_tr, y_tr, current, bs)
        do_log = False
        if i % nn_cfgs['log_intv'] == 0:
            do_log = True
        loss, preds, summary, _ = model.partial_fit(X_batch, y_batch, lr, get_summary=do_log)

        if do_log:
            print('Iteration %d: loss = %f' %(i, loss))
            model.log(summary)
        if current == 0:
            val_loss = val(model, X_val, y_val, bs)
            print('*******************')
            print('Iteration %d: validation loss = %f' %(i, val_loss))
            print('*******************')

    # save model
    model.save(nn_cfgs['tf_sess_path'], cfgs['n_iters'])


