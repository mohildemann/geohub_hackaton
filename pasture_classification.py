from utility_functions import preprocess, split_set, feature_selection, load_data, pca_decomposition
from ml_models import train_keras_model, model_evaluation, rf_model, \
    naive_bayes_model,svm_model, rf_model_parametertuning, svm_model_parametertuning, gradient_boosting_model,\
    keras_hyperparametertuning, ensemple_model
import pandas as pd
 #TODO: rerun parameter opt keras, 90%ellipsoid of points for sampling, check test points locations

train_path = r'data/lcv_pasture_classif.matrix.train_2000..2020_brazil.eumap_summer.school.2022.pq'
val_path = r'data/lcv_pasture_classif.matrix.val_2000..2020_brazil.eumap_summer.school.2022.pq'
test_path = r'data/lcv_pasture_classif.matrix.test_2000..2020_brazil.eumap_summer.school.2022.pq'

merged = load_data(train_path,val_path,test_path)

#preprocessing with feature engineering etc.
preprocessed = preprocess(merged)

#feature selection with user-defined score-threshold
selected_features = feature_selection(preprocessed,method = 'sfe')

#split into training and test (not needed for submission)
Tr_X, Te_X, Tr_Y, Te_Y = split_set(selected_features, rs = 31)

# pca decomposition
decomposed, pca_model = pca_decomposition(Tr_X,Tr_Y, Te_X, Te_Y)

parameter_tuning = False
if parameter_tuning is True:
    rf_model_parametertuning(decomposed)
    #results in max_depth=15, min_samples_split=5, n_estimators=296

    svm_model_parametertuning(decomposed)
    #results in kernel = 'rbf', C=1, gamma=0.005

run_svm = True
run_naive_bayes = True
run_rf = True
run_gradient_boosting = True
run_keras =  False
run_ensemble = True

if run_svm is True:
    svm_model(decomposed)

if run_naive_bayes is True:
    naive_bayes_model(decomposed)

if run_rf is True:
    rf_model(decomposed)

if run_gradient_boosting is True:
    gradient_boosting_model(decomposed)

if run_ensemble is True:
    ensemple_model(decomposed)

if run_keras is True:
    keras_hyperparametertuning(selected_features, nr_epochs=40)
    trained_keras_model = train_keras_model(selected_features, nr_epochs=40, batch_size = 140)


