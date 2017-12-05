import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import librosa as L
from librosa.core import load, get_duration 
import librosa.feature as FT
import time 

_FEATS = [  'mfcc', 'chroma_stft', 'melspectrogram', 'spectral_centroid',
            'spectral_rolloff', 'tonnetz', 'zero_crossing_rate',
         ]
_N_MFCC = 13

class FeatureExtractor():
    def __init__(self, feature_list, feature_pool, l2_norm, sr, wsize, stride, verbose=False):
        for ft_name in feature_list:
            assert ft_name in _FEATS, ('feature %d not supported' %ft_name)
        self.feature_list = feature_list
        self.feature_pool = feature_pool
        self.l2_norm = l2_norm
        self.sr = sr
        self.wsize = wsize
        self.stride = stride
        self.feat_dim = self._calc_all_feat_dim()
        self.spectrogram_shape = self._get_feat_dim('melspectrogram')
        self.verbose = verbose

    def load_song(self, song_fname):
        st = time.time()
        song, sr = load(song_fname, sr=self.sr)
        song = L.effects.trim(song)[0]
        if self.verbose:
            print('Finished loading song "%s" in %f (s)' %(song_fname, time.time()-st))
        return song, sr

    def _get_feat_dim(self, feat_name):
        y, sr = L.load(L.util.example_audio_file(), sr=self.sr)
        window = y[:int(self.wsize * self.sr)]
        feat = self._calc_feat(window, feat_name)
        return feat.shape

    def _calc_all_feat_dim(self):
        return np.sum([self._get_feat_dim(ft)[0] for ft in self.feature_list])

    def _calc_feat(self, window, feat_name):
        feat = None
        # calculate feature
        if feat_name == 'mfcc':
            feat = FT.mfcc(y=window, sr=self.sr, n_mfcc=_N_MFCC)
        elif feat_name == 'chroma_stft':
            feat = FT.chroma_stft(y=window, sr=self.sr)
        elif feat_name == 'melspectrogram':
            feat = FT.melspectrogram(y=window, sr=self.sr, n_mels=128, n_fft=1024, hop_length=512)
            feat = L.power_to_db(feat)
        elif feat_name == 'spectral_centroid':
            feat = FT.spectral_centroid(y=window, sr=self.sr)
        elif feat_name == 'spectral_rolloff':
            feat = FT.spectral_rolloff(y=window, sr=self.sr)
        elif feat_name == 'tonnetz':
            feat = FT.tonnetz(y=window, sr=self.sr)
        elif feat_name == 'zero_crossing_rate':
            feat = FT.zero_crossing_rate(y=window)
        else:
            assert False, 'Invalid feature'

        # pool feature from multiple frames
        if self.feature_pool == 'sum':
            feat = feat.sum(axis=1)
        elif self.feature_pool == 'max':
            feat = feat.max(axis=1)
        elif self.feature_pool == 'mean':
            feat = feat.mean(axis=1)
        elif self.feature_pool == 'flatten':
            feat = feat.flatten()
        elif self.feature_pool == 'none':
            pass
        else:
            assert False, 'Invalid feature pooling scheme'

        # normalize features
        if self.l2_norm and feat.shape[0] > 1:
            feat /= np.linalg.norm(feat)
        return feat

    def _calc_all_feat(self, window):
        feats = list([])
        for ft_name in self.feature_list:
            feat = self._calc_feat(window, ft_name)
            feats.append(feat)
        feats = np.hstack(feats)
        return feats

    def extract_feature(self, song_fname):
        song, _ = self.load_song(song_fname)
        duration = int(get_duration(song))
        n = song.shape[0]
        num_windows = int(np.floor((duration - self.wsize)/ self.stride)) + 20
        feats = np.zeros((num_windows, self.feat_dim), dtype=np.float32)
        start = 0
        counter = 0
        while True:
            end = start + int(self.wsize * self.sr)
            if end >= n:
                break
            window = song[start:end]
            feat_ = self._calc_all_feat(window)
            feats[counter] = feat_
            counter += 1
            start += int(self.stride * self.sr)
            if start >= n:
                break
        if counter < num_windows:
            feats = feats[:counter,:]
        return feats

    def extract_spectrogram(self, song_fname):
        song, _ = self.load_song(song_fname)
        duration = int(get_duration(song))
        n = song.shape[0]
        num_windows = int(np.floor((duration - self.wsize)/ self.stride)) + 20
        feats = np.zeros((num_windows, self.spectrogram_shape[1], self.spectrogram_shape[0]), dtype=np.float32)
        start = 0
        counter = 0
        while True:
            end = start + int(self.wsize * self.sr)
            if end >= n:
                break
            window = song[start:end]
            feat_ = self._calc_feat(window, 'melspectrogram')
            feats[counter] = feat_.T
            counter += 1
            start += int(self.stride * self.sr)
            if start >= n:
                break
        if counter < num_windows:
            feats = feats[:counter,:]
        return feats


if __name__ == '__main__':
    ft = FeatureExtractor(['mfcc'], 22050, 3, 1)
    ft.extract_feature(librosa.util.example_audio_file())
