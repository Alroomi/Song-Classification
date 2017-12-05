import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_gender import gen_config
from models.ann import Classifier 
from models.cnn import CNNClassifier 
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
    return X, y

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
    y_batch = y[start:end,...]
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
    val_acc = 0.0
    while True:
        X_batch, y_batch, current = next_batch(X_tr, y_tr, current, bs)
        loss, acc = model.calc_loss_acc(X_batch, y_batch)
        val_loss += loss 
        val_acc += acc
        counter += 1
        if current == 0:
            break
    val_loss /= counter
    val_acc /= counter
    return val_loss, val_acc

def save_data(fname, data):
    with open(fname, 'wb') as fout:
        pkl.dump(data, fout, pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Invalid arguments'
    task = sys.argv[1].strip()
    model = sys.argv[2].strip()
    cfgs, nn_cfgs = gen_config(task, model)

    if nn_cfgs['model'] == 'ann':
        X, y = load_feat(cfgs['file_all_feat_folder'], cfgs['train_list_fname'])
        model = Classifier(nn_cfgs, nn_cfgs['log_dir'])
    elif nn_cfgs['model'] == 'cnn': 
        X, y = load_feat(cfgs['file_feat_folder'], cfgs['train_list_fname'])
        model = CNNClassifier(nn_cfgs, nn_cfgs['log_dir'])
    else:
        assert False, 'Invalid model type'
        
    if nn_cfgs['zero_mean']:
        mean = X.mean(axis=0, keepdims=True)
        np.save('means/mean_%s_%s' %(cfgs['task'], nn_cfgs['model']), mean)
        X -= mean
    
    print('Finish loading of %d samples' %X.shape[0])

    X, y = shuffle(X,y)
    assert len(np.unique(y)) == nn_cfgs['num_classes'], 'Invalid number of classes'

    y = _to_one_hot(y, nn_cfgs['num_classes'])
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
        loss, acc, summary, _ = model.partial_fit(X_batch, y_batch, lr, get_summary=do_log)

        if do_log:
            print('Iteration %d: loss = %f ; acc = %f' %(i, loss, acc))
            model.log(summary)
        if current == 0:
            val_loss, val_acc = val(model, X_val, y_val, bs)
            print('*******************')
            print('Iteration %d: validation loss = %f ; validation acc = %f' %(i, val_loss, val_acc))
            print('*******************')

    # save model
    model.save(nn_cfgs['tf_sess_path'], cfgs['n_iters'])


