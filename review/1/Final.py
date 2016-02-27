import numpy as np
import pandas
import scipy
import sklearn
import time
import datetime

# 0. Import and prepare data for gradient boosting
features = pandas.read_csv('features.csv', index_col='match_id') 
features_test = pandas.read_csv('features_test.csv', index_col='match_id') 	
	
y_train = features[['radiant_win']] # target (question 4)
x_train = features.drop(features.columns[102:], axis=1) # drop columns after 'duration' should be removed (question 1)
x_test = features_test
#print(x_train.head())
print("Check of shapes")
print(np.shape(x_train))
print(np.shape(x_test))

# check missing values
countval_ind = np.where(x_train.count()-len(x_train) != 0)
print("Columns with empty cells")
print(x_train.columns.values[countval_ind])
# first_blood_time, first_blood_team, first_blood_player1, first_blood_player2, radiant_bottle_time, radiant_courier_time, radiant_flying_courier_time, radiant_first_ward_time, dier_botle_time, dire_courier_time, dire_flying_courier_time, dire_first_ward_time -- команды могли не приобретать предметы bottle или courier в течение первых 5 минут, поэтому это не отражено в данных (question 2)
x_train.fillna(10**(-6), inplace=True) # replacement (question 3)
x_test.fillna(10**(-6), inplace=True)

# 1. Gradient Boosting
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics

scores = pandas.DataFrame(index=range(1),columns=['scores'])
calctime = pandas.DataFrame(index=range(1),columns=['time'])

#learning rate by default = 0.1 # checked different values from [1.0, 0.5, 0.3, 0.2, 0.1]

kf = cross_validation.KFold (n = np.shape(x_train)[0], n_folds = 5, shuffle = True)
num_est = [10, 20, 30] # checked other values as well, still got very bad score on Kaggle
for n in num_est:
	clf_boost = ensemble.GradientBoostingClassifier(n_estimators = n, verbose = True) # max_depth можно ограничить
	start_time = datetime.datetime.now()
	time.sleep(3)
	clf_boost.fit(x_train, y_train)	
	this_time = datetime.datetime.now() - start_time
	these_scores = cross_validation.cross_val_score(clf_boost, x_train, y_train, cv = kf, scoring = 'roc_auc')
	scores[n] = np.mean(these_scores)
	calctime[n] = np.sum(this_time)

print("Scores and time for Gradient Boosting")
from pandas import DataFrame 
print(DataFrame.transpose(scores)) 
print(DataFrame.transpose(calctime)) # for n_estimators = 30 run-time is 35 seconds. 

# 2. Logistic regression

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Regression itself
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
y_train_arr = np.ravel(y_train)

scores = pandas.DataFrame(index=range(1),columns=['scores'])
calctime = pandas.DataFrame(index=range(1),columns=['time'])

kf = cross_validation.KFold (n = np.shape(x_train)[0], n_folds = 5, shuffle = True)
params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] # to select optimal C
for p in params:
	clf_log = linear_model.LogisticRegression(penalty='l2', C = p) # L2 regularization
	start_time = datetime.datetime.now()
	time.sleep(3)
	clf_log.fit(x_train_scaled, y_train_arr)
	this_time = datetime.datetime.now() - start_time
	these_scores = cross_validation.cross_val_score(clf_log, x_train_scaled, y_train_arr, cv = kf, scoring = 'roc_auc')
	scores[p] = np.mean(these_scores)
	calctime[p] = np.mean(this_time)		
print("Scores and time for Logistic Regression")
print(DataFrame.transpose(scores))  # Best score = 0.72
print(DataFrame.transpose(calctime)) # работает быстрее

# Remove categorical variables
x_train = x_train.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero','d1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
x_test = x_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero','d1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Re-estimate the model without categorical variables
params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] 
for p in params:
	clf_log = linear_model.LogisticRegression(penalty='l2', C = p)
	clf_log.fit(x_train_scaled, y_train_arr)
	these_scores = cross_validation.cross_val_score(clf_log, x_train_scaled, y_train_arr, cv = kf, scoring = 'roc_auc')
	scores[p] = np.mean(these_scores)

print("Scores without categorical variables")
print(DataFrame.transpose(scores))  # After removal the score is slightly (in 0.01) better (question 3), because of noise exclusion

# Find # of unique heroes
allheroes = features[['r1_hero','r2_hero', 'r3_hero', 'r4_hero', 'r5_hero','d1_hero','d2_hero', 'd3_hero', 'd4_hero', 'd5_hero']]
unique = np.unique(allheroes)
print("Unique heroes")
print(np.shape(unique)[0]) # Number of unique heroes = 108 (question 4)

# Add heroes 
countheroes = 112 # 108 непустых (использованных в обучающей выборке) героев
x_pick_train = np.zeros((x_train.shape[0], countheroes))
for i, match_id in enumerate(x_train.index):
	for p in range(1,5):
		x_pick_train[i, features.ix[match_id, 'r%d_hero' % p] -1] = 1
		x_pick_train[i, features.ix[match_id, 'd%d_hero' % p] -1] = -1

print("Check sizes for heroes")
print(np.shape(x_pick_train))

x_pick_test = np.zeros((x_test.shape[0], countheroes))
for i, match_id in enumerate(x_test.index):
	for p in range(1,5):
		x_pick_test[i, features_test.ix[match_id, 'r%d_hero' % p] -1] = 1
		x_pick_test[i, features_test.ix[match_id, 'd%d_hero' % p] -1] = -1
print(np.shape(x_pick_test))

x_train_scaled_df = DataFrame(data = x_train_scaled, index = range(np.shape(x_train_scaled)[0]))
x_pick_train_df = DataFrame(data = x_pick_train, index = range(np.shape(x_pick_train)[0]))
x_train_scaled_full = pandas.concat([x_train_scaled_df, x_pick_train_df], axis = 1)
print(np.shape(x_train_scaled_full))

x_test_scaled_df = DataFrame(data = x_test_scaled, index = range(np.shape(x_test_scaled)[0]))
x_pick_test_df = DataFrame(data = x_pick_test, index = range(np.shape(x_pick_test)[0]))
x_test_scaled_full = pandas.concat([x_test_scaled_df, x_pick_test_df], axis = 1)
print(np.shape(x_test_scaled_full))

# Again ee-estimate the model
params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] 
for p in params:
	clf_log = linear_model.LogisticRegression(penalty='l2', C = p)
	clf_log.fit(x_train_scaled_full, y_train_arr)
	these_scores = cross_validation.cross_val_score(clf_log, x_train_scaled_full, y_train_arr, cv = kf, scoring = 'roc_auc')
	scores[p] = np.mean(these_scores)

print("Scores after adding heroes")
print(DataFrame.transpose(scores))  # После добавления "мешка слов" качество улучшилось, мы используем больше информации, которая оказывает значимое влияние на предсказание. Качество на кросс-валидации примерно 0.744

clf_log = linear_model.LogisticRegression(penalty='l2', C = 0.1)
clf_log.fit(x_train_scaled_full, y_train_arr)
predictions = clf_log.predict_proba(x_test_scaled_full)
print("Minumum and maximum values")
print(np.min(predictions))
print(np.max(predictions))

pd = pandas.DataFrame(predictions)
pd.to_csv("predictions.csv")	