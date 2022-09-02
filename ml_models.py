from tensorflow.keras import layers
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
import keras_tuner as kt
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import VotingClassifier

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def rf_model(df):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(max_depth=15, min_samples_split=5, n_estimators=296)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    scores = cross_val_score(clf, df.drop(columns=["class"]), df["class"], cv=10)
    print("Mean and std of F1 score after cross validation with random forest: "+str(scores.mean()) + ', ' + str(scores.std()))

def rf_model_parametertuning(df):
    param_grid = {'max_depth': [3, 5, 10, 15, 20, 25],
                                'min_samples_split': [2, 5, 7, 10],
                                }
    base_estimator = RandomForestClassifier(random_state=0)
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    factor = 2, resource = 'n_estimators',
    max_resources = 300).fit(df.drop(columns=["class"]), df["class"])
    t = sh.best_estimator_
    print(t)
    return sh

def ensemple_model(df, cv = True):
    clf1 = RandomForestClassifier(max_depth=15, min_samples_split=5, n_estimators=296,
                                 random_state=0)
    clf2 = svm.SVC(decision_function_shape='ovo',kernel = 'rbf', C=1, gamma=0.005)
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth = 10)
    eclf = VotingClassifier(
    estimators = [('rf', clf1), ('svm', clf2), ('gnb', clf3)], voting = 'hard')
    if cv is True:
        scores = cross_val_score(eclf, df.drop(columns=["class"]), df["class"], cv=10)
        print("Mean and std of F1 score after cross validation with ensemble: " + str(scores.mean()) + ', ' + str(scores.std()))
    return eclf

def svm_model(df):
    clf = svm.SVC(decision_function_shape='ovo',kernel = 'rbf', C=1, gamma=0.005)
    scores = cross_val_score(clf, df.drop(columns=["class"]), df["class"], cv=10)
    print("Mean and std of F1 score after cross validation with svm: "+str(scores.mean()) + ', ' + str(scores.std()))

def svm_model_parametertuning(df):
    base_estimator = SVC()
    param_grid = {'C': [0.1, 1, 10, 20, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.005, 0.0001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    factor = 2,
    max_resources = 300,
    aggressive_elimination = False,).fit(df.drop(columns=["class"]), df["class"])
    t = sh.best_estimator_
    print(t)
    return sh.best_estimator_

def gradient_boosting_model(df):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth = 10)
    scores = cross_val_score(clf, df.drop(columns=["class"]), df["class"], cv=10)
    print("Mean and std of F1 score after cross validation with gradient boosting: "+str(scores.mean()) + ', ' + str(scores.std()))

def naive_bayes_model(df):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, df.drop(columns=["class"]), df["class"], cv=10)
    print("Mean and std of F1 score after cross validation with gaussian naive bayes: "+str(scores.mean()) + ', ' + str(scores.std()))

def model_evaluation(model, Te_X, Te_Y, verbose = 0):
    #   y_pred = np.argmax(model.predict(Te_X), axis=-1)
    return model.evaluate(Te_X, Te_Y, verbose)
import tensorflow.keras.backend as K
def get_f1(y_true, y_pred):
    # most_probable_class = np.argmax(y_pred, axis=1)
    # f1_score(y_true, most_probable_class, average='macro')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def train_keras_model(df, model = None, nr_epochs = 10, batch_size = 32):
    train, val, test = np.split(df.sample(frac=1), [int(0.7 * len(df)), int(0.9 * len(df))])

    def baseline_model():
        def get_f1(y_true, y_pred):  # taken from old keras source code
            t = f1_score(y_true, y_pred)
            return t
        # Create model here
        model = Sequential()
        model.add(Dense(15, input_dim=df.shape[1]-1, activation='relu'))  # Rectified Linear Unit Activation Function
        model.add(Dense(15, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(3, activation='softmax'))  # Softmax for multi-class classification
        # Compile model here
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[ 'accuracy'])
        return model
    if model is None:
    # Create Keras Classifier and use predefined baseline model
        estimator = KerasClassifier(build_fn=baseline_model, epochs=nr_epochs, batch_size=batch_size, verbose=0)
    else:
        estimator = KerasClassifier(build_fn=model, epochs=nr_epochs, batch_size=batch_size, verbose=0)
    X_test = test.drop('class', axis=1).values
    Y_test = test['class'].values
    X_train = train.drop('class', axis=1).values
    Y_train = train['class'].values
    estimator.fit(X_train, Y_train)
    pred = estimator.model.predict(X_test)
    most_probable_class = np.argmax(pred, axis=1) + 1
    print("F1 from keras NN from manual architecture + parameters: %f" % f1_score(Y_test, most_probable_class, average = 'macro'))

def keras_hyperparametertuning(df, nr_epochs):
    f_scores = []
    for i in range(10):
        train, val, test = np.split(df.sample(frac=1, random_state = i), [int(0.8 * len(df)), int(0.9 * len(df))])
        X_test = test.drop('class', axis=1).values
        Y_test = test['class'].values
        X_train = train.drop('class', axis=1).values
        Y_train = train['class'].values
        Y_test_copy = Y_test.copy()


        def model_builder(hp):
            model = Sequential()
            model.add(Dense(15, input_dim=df.shape[1]-1, activation='relu'))
            for i in range(hp.Int("num_layers", 1, 4)):
                model.add(
                    layers.Dense(
                        # Tune number of units separately.
                        units=hp.Int(f"units_{i}", min_value=10, max_value=400, step=10),
                        activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
                    )
                )
            if hp.Boolean("dropout"):
                model.add(layers.Dropout(rate=0.25))
            model.add(Dense(4, activation='softmax'))
            learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=[get_f1],
            )
            return model

        tuner = kt.Hyperband(model_builder,  # the hypermodel
                             objective=kt.Objective("val_get_f1", direction="max"),  # objective to optimize
                             max_epochs=nr_epochs,
                             factor=3,  # factor which you have seen above
                             directory='dir',  # directory to save logs
                             project_name='khyperband')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Perform hypertuning

        Y_train = to_categorical(Y_train, 4)
        Y_test = to_categorical(Y_test, 4)
        tuner.search(X_train, Y_train, epochs=nr_epochs, validation_split=0.2, callbacks=[stop_early], verbose=False)
        best_hps = tuner.get_best_hyperparameters()[0]

        print(tuner.results_summary())
        hypermodel = tuner.hypermodel.build(best_hps)
        history = hypermodel.fit(X_train, Y_train, epochs=nr_epochs, validation_split=0.3)
        eval_result = hypermodel.evaluate(X_test, Y_test)
        pred = hypermodel.predict(X_test)

        #pred = estimator.model.predict(X_test)
        most_probable_class = np.argmax(pred, axis=1)
        f_score = f1_score(Y_test_copy, most_probable_class, average='macro')
        f_scores.append(f_score)
    f_scores = np.array(f_scores)
    print("F1 from keras NN with hyperparametertuning with mean of " + str(np.mean(f_scores)) + " and stf of "+ str(np.std(f_scores)))