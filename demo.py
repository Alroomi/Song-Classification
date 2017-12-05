import os, os.path, sys
import h5py 
import numpy as np
import pickle as pkl
from configs_genre import gen_config as genre_config
from configs_gender import gen_config as gender_config
from configs_year import gen_config as year_config
from models.ann import Classifier 
from models.cnn import CNNClassifier
from feature_extractor import FeatureExtractor
from sklearn.utils.extmath import softmax
from utils import parse_csv_list_file

def load_models(task):
    if task.startswith('Year'):
        cfgs_1, nn_cfgs_1 = year_config(task, 'ann', 'decade')
        cfgs_2, nn_cfgs_2 = year_config(task, 'cnn', 'decade')
    elif task.startswith('Genre'):
        cfgs_1, nn_cfgs_1 = genre_config(task, 'ann')
        cfgs_2, nn_cfgs_2 = genre_config(task, 'cnn')
    elif task.startswith('Gender'):
        cfgs_1, nn_cfgs_1 = gender_config(task, 'ann')
        cfgs_2, nn_cfgs_2 = gender_config(task, 'cnn')

    ann = Classifier(nn_cfgs_1, log_dir=None)
    ann.restore('%s-%d' %(nn_cfgs_1['tf_sess_path'], cfgs_1['n_iters']))
    cnn = CNNClassifier(nn_cfgs_2, log_dir=None)
    cnn.restore('%s-%d' %(nn_cfgs_2['tf_sess_path'], cfgs_2['n_iters']))
    return ann, cnn

def init_ft(task):
    if task.startswith('Year'):
        cfgs, nn_cfgs = year_config(task, 'ann', 'decade')
    elif task.startswith('Genre'):
        cfgs, nn_cfgs = genre_config(task, 'ann')
    elif task.startswith('Gender'):
        cfgs, nn_cfgs = gender_config(task, 'ann')

    ft1 = FeatureExtractor(cfgs['feature_list'], cfgs['feature_pool'], cfgs['l2_norm'], cfgs['sr'], 1, cfgs['stride'])
    ft2 = FeatureExtractor(['melspectrogram'], 'none', cfgs['l2_norm'], cfgs['sr'], 3, cfgs['stride'])
    return ft1, ft2

def get_utils(task):
    if task.startswith('Year'):
        cfgs, nn_cfgs = year_config(task, 'ann', 'decade')
    elif task.startswith('Genre'):
        cfgs, nn_cfgs = genre_config(task, 'ann')
    elif task.startswith('Gender'):
        cfgs, nn_cfgs = gender_config(task, 'ann')
    _, cls_to_id, id_to_cls = parse_csv_list_file(cfgs['train_list_fname'])
    ann_mean = np.load('means/mean_%s_%s.npy' %(task,'ann'))
    cnn_mean = np.load('means/mean_%s_%s.npy' %(task,'cnn'))
    return cls_to_id, id_to_cls, ann_mean, cnn_mean

def build_label_map_year2decade(cls_to_id):
    new_cls_to_id = dict({})
    new_id_to_cls = dict({})
    old_to_new_ids = dict({})
    counter = 0
    years = cls_to_id.keys()
    years = sorted(years)
    for year in years:
        decade = '%d0s' %int(np.floor(int(year) / 10))
        if decade not in new_cls_to_id:
            new_cls_to_id[decade] = counter # '19x0s' -> 0 ... 
            new_id_to_cls[counter] = decade # 0 -> '19x0s'
            counter += 1
        old_to_new_ids[cls_to_id[year]] = new_cls_to_id[decade]
    num_ids = len(new_id_to_cls.keys())
    return old_to_new_ids, new_id_to_cls, num_ids 

def sum_score(scores):
    sum_scores = scores.sum(axis=0)
    final_pred = np.argmax(sum_scores)
    return final_pred

def ensemble(scores1, scores2):
    ss1 = scores1.sum(axis=0).reshape([1,-1])
    ss2 = scores2.sum(axis=0).reshape([1,-1])
    ss1 = softmax(ss1)
    ss2 = softmax(ss2)
    final_scores = ss1 + ss2 
    final_pred = np.argmax(final_scores)
    return final_pred

def predict_genre(feat1, feat2, ann, cnn, cls_to_id, id_to_cls, mean1, mean2):
    # ann
    preds1, scores1 = ann.predict(feat1 - mean1)
    final_pred1 = sum_score(scores1)
    # cnn
    preds2, scores2 = cnn.predict(feat2 - mean2)
    final_pred2 = sum_score(scores2)
    # ensemble
    ensemble_pred = ensemble(scores1, scores2)

    print('--------------Genre Prediction--------------')
    print('FeatureExtraction1: %s' %id_to_cls[final_pred1])
    print('FeatureExtraction2: %s' %id_to_cls[final_pred2])
    print('Ensemble: %s' %id_to_cls[ensemble_pred])

def predict_gender(feat1, feat2, ann, cnn, cls_to_id, id_to_cls, mean1, mean2):
    # ann
    preds1, scores1 = ann.predict(feat1 - mean1)
    final_pred1 = sum_score(scores1)
    # cnn
    preds2, scores2 = cnn.predict(feat2 - mean2)
    final_pred2 = sum_score(scores2)
    # ensemble
    ensemble_pred = ensemble(scores1, scores2)

    print('--------------Gender Prediction--------------')
    print('FeatureExtraction1: %s' %id_to_cls[final_pred1])
    print('FeatureExtraction2: %s' %id_to_cls[final_pred2])
    print('Ensemble: %s' %id_to_cls[ensemble_pred])

def predict_year(feat1, feat2, ann, cnn, cls_to_id, id_to_cls, mean1, mean2):
    # ann
    preds1, scores1 = ann.predict(feat1 - mean1)
    final_pred1 = sum_score(scores1)
    # cnn
    preds2, scores2 = cnn.predict(feat2 - mean2)
    final_pred2 = sum_score(scores2)
    # ensemble
    ensemble_pred = ensemble(scores1, scores2)

    print('--------------Year Prediction--------------')
    print('FeatureExtraction1: %s (%s)' %(id_to_cls[final_pred1], id_to_cls[final_pred1]))
    print('FeatureExtraction2: %s (%s)' %(id_to_cls[final_pred2], id_to_cls[final_pred2]))
    print('Ensemble: %s (%s)' %(id_to_cls[ensemble_pred], id_to_cls[ensemble_pred]))

if __name__ == '__main__':
    len(sys.argv) == 2, 'Path to folder containing the songs need to be provided'
    song_folder = sys.argv[1]
    # ./data/6th Nov/genre/songs

    genre_ann, genre_cnn = load_models('Genre_fold_1')
    genre_cls_to_id, genre_id_to_cls, genre_ann_mean, genre_cnn_mean = get_utils('Genre_fold_1')
    ft1, ft2 = init_ft('Genre_fold_1')

    gender_ann, gender_cnn = load_models('Gender_fold_1')
    gender_cls_to_id, gender_id_to_cls, gender_ann_mean, gender_cnn_mean = get_utils('Gender_fold_1')

    year_ann, year_cnn = load_models('Year_fold_1')
    year_cls_to_id, year_id_to_cls, year_ann_mean, year_cnn_mean = get_utils('Year_fold_1')
    year_cls_to_id, year_id_to_cls, _ = build_label_map_year2decade(year_cls_to_id)

    filenames = os.listdir(song_folder)
    for fname in filenames:
        fname = fname.strip()
        if not fname.endswith('.mp3'):
            continue
        print('--------------------------------------------')
        print(fname)
        print('Extracting Feature 1 ...')
        feat1 = ft1.extract_feature(os.path.join(song_folder, fname))
        print('Extracting Feature 2 ...')
        feat2 = ft2.extract_spectrogram(os.path.join(song_folder, fname))

        print('Done.')
        predict_genre(feat1, feat2, genre_ann, genre_cnn, genre_cls_to_id, genre_id_to_cls, genre_ann_mean, genre_cnn_mean)
        predict_gender(feat1, feat2, gender_ann, gender_cnn, gender_cls_to_id, gender_id_to_cls, gender_ann_mean, gender_cnn_mean)
        predict_year(feat1, feat2, year_ann, year_cnn, year_cls_to_id, year_id_to_cls, year_ann_mean, year_cnn_mean)
        input("Press Enter to continue...")
