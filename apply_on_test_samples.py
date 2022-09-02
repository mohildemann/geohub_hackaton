import pyarrow.parquet as pq
import pandas as pd
from utility_functions import preprocess, feature_selection, split_set, pca_decomposition
from ml_models import ensemple_model

#run best model to predict test points
new_train_path = r'data/new_data/lcv_pasture_classif.matrix.train_2000..2020_brazil.eumap_summer.school.2022.csv'
new_val_path = r'data/new_data/lcv_pasture_classif.matrix.val_2000..2020_brazil.eumap_summer.school.2022.csv'
new_test_path = r'data/new_data/lcv_pasture_classif.matrix.test_2000..2020_brazil.eumap_summer.school.2022.csv'

new_train_samples = pd.read_csv(new_train_path)
new_val_samples = pd.read_csv(new_val_path)
new_test_samples = pd.read_csv(new_test_path)

merged = pd.concat([new_train_samples, new_val_samples])
merged_wo_class = merged

preprocessed = preprocess(merged)

#feature selection with user-defined score-threshold
selected_features = feature_selection(preprocessed,method = 'sfe')

#split into training and test (not needed for submission)
Tr_X, Te_X, Tr_Y, Te_Y = split_set(selected_features, rs = 31)

# pca decomposition
decomposed, pca_model = pca_decomposition(Tr_X,Tr_Y, Te_X, Te_Y)
ensemble_model = ensemple_model(decomposed, cv = False)
ensemble_model.fit(decomposed.drop(columns=["class"]), decomposed["class"])

X_test_pca = pca_model.transform(Te_X)
new_test_samples_decomposed = pd.DataFrame(X_test_pca)
#new_test_samples_decomposed = pca_model.transform(new_test_samples)

test_pred = ensemble_model.predict(new_test_samples_decomposed)
result = pd.DataFrame({ 'pred':test_pred, 'id': Te_Y.index })
result.to_csv(r'results/moritz_hildemann.csv')