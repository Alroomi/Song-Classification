# Song Classification
README

Project: 	
Classification and Regression of Songs
1.	Prequisite
In this project, I use Python version 3.5 for my implementations. 
In order to run the code and produce results as in the report, the following libraries need to be installed. 

- librosa (tested with v0.5.1)
- sklearn (tested with v0.19.0)
- numpy (tested with v1.13.1)
- h5py (tested with v2.6.0)
- tensorflow (tested with v1.2 and v1.3.0)
The code has been tested in Linux (Ubuntu 14.04) and Mac OSX and Windows. 
2.	Folder Structure
2.1.	Data
All the data needs to be placed inside the folder data. 
For example, to work with the dataset collected until 6th November, the data folder structure should look like below
├── data
│   ├── 6th Nov	
│       ├── genre
│           └── songs
│               └── .csv files
│               └── file_all_feat
│               └── file_spectrogram
│       ├── gender
│           └── songs
│               └── .csv files
│               └── file_all_feat
│               └── file_spectrogram
│       ├── year
│           └── songs
│               └── .csv files
│               └── file_all_feat
│               └── file_spectrogram
 
-	In each task folder (i.e. genre, gender, year), there are several .csv files. These .csv files define different lists of songs for training and testing (5 folds) and the corresponding labels. 
-	To work directly from the song files, the files need to be placed insides the songs folders. 
-	The feature files extracted using FeatureExtraction1 method need to be placed inside the file_all_feat folders. Each feature file corresponds to one song file, and is in .npz format in Numpy. 
-	The feature files extracted using FeatureExtraction2 method need to be placed inside the file_spectrogram folders. Each feature file corresponds to one song file, and is in .npz format in Numpy.

2.2. Models 
The trained models are saved inside the corresponding folder for each experiment. For example, the training results of Year_fold_1 are saved in output_Year_fold_1. 

The test results are saved inside results folder. The result for each experiment is saved as .csv file and the naming corresponds to the experiment, e.g. result_Year_fold_1_ann.csv (here ann means the result is with the feature extraction 1 and ANN model). 
3.	Quick Run	
To avoid spending a lot of time on feature extraction and model training, you can directly run testing using the provided trained models and feature files, with the default configurations. 

a)	After preparing the data in Step 2.1, you can run the testing code for classification using single models as for the tasks with the following syntax:
python test_genre.py <task> <model>
python test_gender.py <task> <model>
python test_year.py <task> <model> <label_type>
where:
-	task: the fold to train/test on 
-	model: ‘ann’ or ‘cnn’ (corresponding to FeatureExtraction1 and FeatureExtraction2 respectively)
-	label_type: this option is only available for working with the year data. You can specify ‘decade’ or ‘year’, to perform classification with the decade labels or the year labels. 

For example:
-	python test_genre.py Genre_fold_1 cnn: run test on genre data fold 1 using FeatureExtraction2 and CNN model.
-	python test_gender.py Gender_fold_1 ann: run test on gender data fold 1 using FeatureExtraction1 and ANN model.
-	python test_year.py Year_fold_1 ann decade: run test on year data fold 1 using FeatureExtraction1 and ANN model, with the original years transformed into decades.

All results are saved to the results folder in .csv file format. Each .csv file contains the results for each song in the test lists. 

b)	To test the ensemble of FeatureExtraction1 and FeatureExtraction2 results, you can use the following commands:
python test_genre_ensemble.py <task> <model>
python test_gender_ensemble.py <task> <model>
python test_year_ensemble.py <task> <model> <label_type>

c)	To test the regression model on the year data, you can use the following command:
python test_year_regressor.py <task> 
where:
-	task: the fold to train/test on 
For example:
-	python test_ year_regressor.py Year_fold_1

d)	Fusion of models’ results
An experiment of fusion a decision level between 3 models is carried out in this project. 
In order to run this test, you can use the following commands:
	cd results/fusion
	python fuse.py <task> <mode>
where:
-	task: ‘Genre_fold_1’ or ‘Gender_fold_1’ or ‘Year_fold_1’
-	mode: fusion model, choose from:
o	vote: majority voting
o	score_weight: weighting based on model’s scores
o	pfm: pairwise fusion matrix based fusion
The result will be saved in .csv file inside the same folder. 

e) Demo
To run the live demo, you can use the following commands:
	python demo.py <song_folder>
where:
-	song_folder: the path to the folder containing the songs to do predictions on
For example:
         python demo.py ./data/6th\ Nov/year/songs

Note: the code will loads all the .mp3 files inside the specified folder to run prediction on. 
4.	Running from Scratch
To run everything from scratch, please follow the instruction below:

4.1.	Feature Extraction 
4.1.1.	FeatureExtraction1
-	python prepare_feature_data.py genre: extract features (type 1) for the songs for the genre task
-	python prepare_feature_data.py gender: extract features (type 1) for the songs for the gender task
-	python prepare_feature_data.py year: extract features (type 1) for the songs for the year task

The function will load the lists of files from Genre.csv or Gender.csv or Year.csv from ./data/6th Nov/<task>/songs, where <task> is genre, gender or year respectively. 

The output feature files are saved into ./data/6th Nov/<task>/songs/file_all_feat.

4.1.2.	FeatureExtraction2
-	python prepare_feature_data_spectrogram.py genre: extract features (type 2) for the songs for the genre task
-	python prepare_feature_data_spectrogram.py gender: extract features (type 2) for the songs for the gender task
-	python prepare_feature_data_spectrogram.py year: extract features (type 2) for the songs for the year task

The function will load the lists of files from Genre.csv or Gender.csv or Year.csv from ./data/6th Nov/<task>/songs, where <task> is genre, gender or year respectively. 

The output feature files are saved into ./data/6th Nov/<task>/songs/file_spectrogram.

4.2.	Genre Classification
Training: python train_genre.py <task> <model>
Testing:  python test_genre.py <task> <model>
Testing Ensemble: python test_genre_ensemble.py <task> <model>
where:
-	task: the fold to train/test on 
-	model: ‘ann’ or ‘cnn’ (corresponding to FeatureExtraction1 and FeatureExtraction2 respectively)
For example: 
python train_genre.py Genre_fold_1 cnn 
python test_genre.py Genre_fold_1 cnn
run train and test on genre data fold 1 using FeatureExtraction2 and CNN model. 
4.3.	Gender Classification
Training: python train_gender.py <task> <model>
Testing:  python test_gender.py <task> <model>
Testing Ensemble: python test_gender_ensemble.py <task> <model>
where:
-	task: the fold to train/test on 
-	model: ‘ann’ or ‘cnn’ (corresponding to FeatureExtraction1 and FeatureExtraction2 respectively)
For example: 
python train_gender.py Gender_fold_1 ann
python test_gender.py Gender_fold_1 ann
run train and test on gender data fold 1 using FeatureExtraction1 and ANN model. 

Note:
Besides the file normal folds, there are other tasks which are available:
-	Gender_f_m_fold_1 (to fold 5): gender data with the two labels, ‘f’ and ‘m’
-	Gender_s_m_fold_1 (to fold 5): gender data with the two labels, ‘s’ and ‘m’ (single and multiple respectively)

4.4.	Year Classification 
Training: python train_year.py <task> <model> <label_type>
Testing:  python test_year.py <task> <model> <label_type>
Testing Ensemble: python test_year_ensemble.py <task> <model> <label_type>
where:
-	task: the fold to train/test on 
-	model: ‘ann’ or ‘cnn’ (corresponding to FeatureExtraction1 and FeatureExtraction2 respectively)
-	label_type: type of label, e.g. decade or year.
For example: 
python train_year.py Year_fold_1 ann decade
python test_year.py Year_fold_1 ann decade
run train and test on year data fold 1 using FeatureExtraction1 and ANN model, with the original year labels transformed into decades.

4.5.	Year Regression
Training: python train_year_regressor.py <task> 
Testing:  python test_year_regressor.py <task> 
where:
-	task: the fold to train/test on 
For example: 
python train_year_regressor.py Year_fold_1 
python test_year_regressor.py Year_fold_1
run train and test for the regression task on year data fold 1 using FeatureExtraction2 and CNN model.  

Note:
The trained models are saved into output folders and the test results are saved into results folders as specified in section 2.2.


