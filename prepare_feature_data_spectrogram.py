import os, sys
import h5py 
import numpy as np
from configs_genre import gen_config as genre_config
from configs_gender import gen_config as gender_config
from configs_year import gen_config as year_config
from feature_extractor import FeatureExtractor

def prepare_train_data(data_dir, list_fname, ft, outfname):
    print('Writing data to %s' %outfname)
    fin = open(list_fname, 'r')
    lines = fin.readlines()
    n_lines = len(lines)
    feat_shape = ft.spectrogram_shape
    ite = 0
    n_skip_files = 0
    skip_name_fout = open('skipped_files_train.txt', 'w')
    with h5py.File(outfname, 'w') as f:
        feat_dset = f.create_dataset('feature', (n_lines * 200, feat_shape[1], feat_shape[0]), 
            maxshape=(None, feat_shape[1], feat_shape[0]), chunks=True, compression="gzip")
        label_dset = f.create_dataset('label', (n_lines * 200,), 
            maxshape=(None, ), chunks=True, dtype='i8', compression="gzip")
        db_counter = 0
        batch_counter = 0
        for line in lines:
            fname, label = line.strip().split('::')
            print(fname)
            if not os.path.exists(os.path.join(data_dir, fname)):
                print('File not found...Skip!')
                skip_name_fout.write('%s\n' %fname)
                n_skip_files += 1
                continue
            feats = ft.extract_spectrogram(os.path.join(data_dir, fname))
            n_feats = feats.shape[0]
            feat_dset[ite:ite+n_feats, ...] = feats
            label_dset[ite:ite+n_feats] = np.repeat([int(label.strip())], n_feats)
            ite += n_feats
            print('Save %d samples to db' %(n_feats))
            current_size = feat_dset.shape[0]
            if ite >= current_size - n_lines * 50:
                new_size = current_size + n_lines * 200
                feat_dset.resize(new_size, axis=0)
                label_dset.resize(new_size, axis=0)
                print('Extend db from %d to %d' %(current_size, new_size))

        current_size = feat_dset.shape[0]   
        feat_dset.resize(ite, axis=0)
        label_dset.resize(ite, axis=0)
        print('Shrink db from %d to %d' %(current_size, ite))
    print('%s files skipped' %n_skip_files)
    skip_name_fout.close()
    fin.close()

def prepare_test_data(data_dir, list_fname, ft, out_dir):
    fin = open(list_fname, 'r')
    lines = fin.readlines()
    n_lines = len(lines)
    ite = 0
    n_skip_files = 0
    if not os.path.isdir(os.path.join(out_dir)):
        os.makedirs(os.path.join(out_dir))

    skip_name_fout = open('skipped_files_test.txt', 'w')
    for line in lines:
        fname, label = line.strip().split('::')
        print(fname)
        if not os.path.exists(os.path.join(data_dir, fname)):
            print('File not found...Skip!')
            n_skip_files += 1
            skip_name_fout.write('%s\n' %(fname))
            continue
        feats = ft.extract_spectrogram(os.path.join(data_dir, fname))
        np.save(os.path.join(out_dir, fname.replace('.mp3','')), feats)
    fin.close()
    skip_name_fout.close()
    print('%d files skipped' %n_skip_files)

def prepare_file_data(data_dir, list_fname, ft, out_dir):
    fin = open(list_fname, 'r')
    lines = fin.readlines()
    n_lines = len(lines)
    ite = 0
    n_skip_files = 0
    if not os.path.isdir(os.path.join(out_dir)):
        os.makedirs(os.path.join(out_dir))

    skip_name_fout = open('skipped_files.txt', 'w')
    for line in lines:
        fname, label = line.strip().split('::')
        print(fname)
        if not os.path.exists(os.path.join(data_dir, fname)):
            print('File not found...Skip!')
            n_skip_files += 1
            skip_name_fout.write('%s\n' %(fname))
            continue
        feats = ft.extract_spectrogram(os.path.join(data_dir, fname))
        np.save(os.path.join(out_dir, fname.replace('.mp3','')), feats)
    fin.close()
    skip_name_fout.close()
    print('%d files skipped' %n_skip_files)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        task = 'genre'
    else:
        task = sys.argv[1]
    if task == 'genre':
        cfgs, nn_cfgs = genre_config(task, 'cnn')
    elif task == 'gender':
        cfgs, nn_cfgs = gender_config(task, 'cnn')
    elif task == 'year':
        cfgs, nn_cfgs = year_config(task, 'cnn', 'decade')
    else:
        assert False, 'Invalid task'
    
    ft = FeatureExtractor(['melspectrogram'], 'none', cfgs['l2_norm'],
        cfgs['sr'], cfgs['wsize'], cfgs['stride'])
    prepare_file_data(cfgs['data_dir'], cfgs['list_fname'], ft, cfgs['file_feat_folder'])
