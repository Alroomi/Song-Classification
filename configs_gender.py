import os 

def gen_config(task, model):
    cfgs = dict({})

    # configs for data preparation
    cfgs['task'] = task
    cfgs['data_dir'] = './data/6th Nov/gender/songs'
    cfgs['train_list_fname'] = os.path.join(cfgs['data_dir'], '%s_train.csv' %cfgs['task'])
    cfgs['test_list_fname'] = os.path.join(cfgs['data_dir'], '%s_test.csv' %cfgs['task'])
    cfgs['list_fname'] = os.path.join(cfgs['data_dir'], 'temp_gender.txt')

    cfgs['sr'] = 22050  # sampling rate
    cfgs['wsize'] = 3   # window size (used to extract feature) (in seconds)
    cfgs['stride'] = 1  # stride of sliding window (in seconds)
    cfgs['feature_list'] = ['mfcc', 'chroma_stft', 'melspectrogram', 'spectral_centroid',
        'spectral_rolloff', 'tonnetz', 'zero_crossing_rate']
    cfgs['feature_pool'] = 'mean' # 'sum', 'flatten', 'max'
    cfgs['l2_norm'] = False

    # training configuration
    cfgs['train_h5'] = os.path.join(cfgs['data_dir'], 'train_spectrogram.h5')
    cfgs['test_feat_folder'] = os.path.join(cfgs['data_dir'], 'test_spectrogram')
    cfgs['file_feat_folder'] = os.path.join(cfgs['data_dir'], 'file_spectrogram')
    cfgs['file_all_feat_folder'] = os.path.join(cfgs['data_dir'], 'file_all_feat')
    cfgs['n_iters'] = 20000

    # output 
    cfgs['base_output'] = './outputs/output_%s' %cfgs['task']

    # ANN/CNN configs
    nn_cfgs = dict({})
    nn_cfgs['model'] = model # 'ann' or 'cnn'
    if nn_cfgs['model'] == 'ann':
        cfgs['base_output'] += '/ann'
    elif nn_cfgs['model'] == 'cnn':
        cfgs['base_output'] += '/cnn'
    else:
        assert False, 'Invalid model type'
    nn_cfgs['h'] = 130
    nn_cfgs['w'] = 128
    nn_cfgs['n_filters'] = [32, 32, 64]
    nn_cfgs['kernel_sizes'] = [[1, 128], [1, 1], [1, 1]]
    nn_cfgs['strides'] = [[1, 128], [1, 1], [1, 1]]
    nn_cfgs['num_classes'] = 5 # for all genders
    nn_cfgs['inp_dim'] = 162
    nn_cfgs['hidden_sizes'] = [1024, 512]
    nn_cfgs['keep_prob'] = 0.5
    nn_cfgs['weight_decay'] = 1e-4
    nn_cfgs['use_bn'] = True
    nn_cfgs['initial_lr'] = 1e-3
    nn_cfgs['tf_sess_path'] = os.path.join(cfgs['base_output'], 'snapshot/')
    nn_cfgs['log_dir'] = os.path.join(cfgs['base_output'], 'log/')
    nn_cfgs['bs'] = 128
    nn_cfgs['log_intv'] = 10
    nn_cfgs['zero_mean'] = True

    # testing configs
    nn_cfgs['ensemble_mode'] = 'score_sum' # 'voting' or 'score_sum'
    return cfgs, nn_cfgs
