# GradientBoostingClassifier
import datetime
import pandas as pd
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

features = pd.read_csv('./data/features.csv', index_col='match_id')
features_test_header = pd.read_csv('./data/features_test.csv', index_col='match_id', nrows=1)

X_train = features[features_test_header.columns]

# 1. Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
counts = X_train.count()
columns_name_with_missing_names = []
for column_name in X_train.columns:
    if counts[column_name] < X_train.shape[0]:
        columns_name_with_missing_names.append(column_name)
print columns_name_with_missing_names
# ['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time',
# 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']

X_train.fillna(0, inplace=True, axis='columns')

# 2. Как называется столбец, содержащий целевую переменную?
y_train = features['radiant_win']

kf = KFold(len(y_train), n_folds=5, shuffle=True, random_state=0)

# 3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти выше по тексту.
# Какое качество при этом получилось?
start_time = datetime.datetime.now()
clf = GradientBoostingClassifier(n_estimators=30)
score = cross_val_score(clf, X_train, y_train, cv=kf, scoring='roc_auc').mean()
print 'time elapsed:', datetime.datetime.now() - start_time  # 0:02:49.536002
print 'score:', score  # 0.688614863733

# 4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что можно сделать, чтобы ускорить его обучение при увеличении количества деревьев?
clf = GradientBoostingClassifier()
grid = {'n_estimators': [10, 20, 30, 100], 'learning_rate': [1, 0.5, 0.3, 0.2, 0.1]}
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1, verbose=2)
gs.fit(X_train, y_train)
print 'best estimator:', gs.best_estimator_
print 'best n_estimators:', gs.best_params_['n_estimators']  # 100
print 'best score:', gs.best_score_  # 0.713737966559


# LogisticRegression
import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('./data/features.csv', index_col='match_id')
features_test_header = pd.read_csv('./data/features_test.csv', index_col='match_id', nrows=1)

X_train = features[features_test_header.columns]
X_train.fillna(0, inplace=True, axis='columns')
y_train = features['radiant_win']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
kf = KFold(len(y_train), n_folds=5, shuffle=True, random_state=1)

# 1. Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга?
# Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
clf = LogisticRegression(penalty='l2')
grid = {'C': np.power(10.0, np.arange(-5, 6))}
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1, verbose=2)
gs.fit(X_train_scaled, y_train)
print 'best estimator:', gs.best_estimator_
print 'best score:', gs.best_score_  # 0.716375794854

start_time = datetime.datetime.now()
score = cross_val_score(gs.best_estimator_, X_train_scaled, y_train, cv=kf, scoring='roc_auc').mean()
print 'time elapsed:', datetime.datetime.now() - start_time  # 0:00:18.243521
print 'score:', score  # 0.716375794854

# 2. Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)?
# Чем вы можете объяснить это изменение?
columns_exclude = ['lobby_type'] + [team + str(i) + '_hero' for team in ['r', 'd'] for i in xrange(1, 6)]
X_train_filtered = X_train[X_train.columns.difference(columns_exclude)]
X_train_scaled = scaler.fit_transform(X_train_filtered)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1, verbose=2)
gs.fit(X_train_scaled, y_train)
print 'best estimator:', gs.best_estimator_
print 'best score:', gs.best_score_  # 0.716408868155

# 3. Сколько различных идентификаторов героев существует в данной игре?
heroes_index = [t + str(i) + '_hero' for t in ['r', 'd'] for i in xrange(1, 6)]
heroes = X_train[heroes_index]
N = len(pd.unique(heroes.values.ravel()))
print N  # 108

# 4. Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом?
# Чем можно это объяснить?
X_pick = np.zeros((heroes.shape[0], heroes.values.max()))
for i, match_id in enumerate(heroes.index):
    for p in xrange(5):
        X_pick[i, heroes.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, heroes.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_train_with_heroes = np.c_[X_train_scaled, X_pick]

gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1, verbose=2)
gs.fit(X_train_with_heroes, y_train)
print 'best estimator:', gs.best_estimator_
print 'best score:', gs.best_score_  # 0.751873146027

# 5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
X_test = pd.read_csv('./data/features_test.csv', index_col='match_id')
X_test.fillna(0, inplace=True, axis='columns')
X_test_filtered = X_test[X_test.columns.difference(columns_exclude)]
X_test_scaled = scaler.fit_transform(X_test_filtered)
heroes = X_test[heroes_index]
X_pick = np.zeros((heroes.shape[0], heroes.values.max()))
for i, match_id in enumerate(heroes.index):
    for p in xrange(5):
        X_pick[i, heroes.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, heroes.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_test_with_heroes = np.c_[X_test_scaled, X_pick]
predict = gs.best_estimator_.predict_proba(X_test_with_heroes)[:, 1]
print 'min:', predict.min()  # 0.00858052299709
print 'max:', predict.max()  # 0.996459245435
