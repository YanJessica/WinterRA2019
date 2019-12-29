from DataProcess import DataPreProcessing
from sklearn import preprocessing
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from static_funcs import *


class Model:

    def __init__(self, x_var=True, test_var=False, split_rates=(0.7, 0.15, 0.15)):
        self.x_var, self.test_var, self.split_rates = x_var, test_var, split_rates  # split_rates: train, valid, test
        self.data_processing = None
        self.df_all = None
        self.train_len, self.valid_len, self.test_len = None, None, None
        self.x_scalar, self.y_scalar = None, None
        self.data_features, self.patnums_idates, self.x_matrix, self.y_vector = None, None, None, None
        self.dict_performance = {'processor_name': [], 'scalar_x': [], 'scalar_y': [],
                                 'reg_name': [], 'reg': [], 'hyperparameter_setting': [],
                                 'mae_scale': [], 'mae': []}

        self._read_data(filename='data/data_process.obj')
        self._x_variables_process()
        self._data_arrange()

    def _read_data(self, filename='data/data_process.obj'):
        """
        @param filename: str
        """
        with open(filename, 'rb') as file:
            self.data_processing = pickle.load(file)

    def _x_variables_process(self):
        """
        Module One: non-text variables processing
        """
        # 1. data splits
        df_train, df_valid, df_test = self._data_split_rates()

        # 2. feature engineering after data splits
        # 2.1 fill missing values ('figures#num_figures', 'figures#num_sheets') based on class.
        for col_name in ['figures#num_figures', 'figures#num_sheets']:
            df_train, df_valid, df_test = data_missing_fill(df_train, df_valid, df_test, col_name)

        # 2.1 cited patents - only count ones before the splitting date. Note: Patents issued later have larger patnums
        df_train, df_valid, df_test = n_cited_calculation(self.df_all, df_train, df_valid, df_test)

        # 2.2 class and subclass
        self.df_all = class_subclass_processing(df_train, df_valid, df_test)

    def _data_split_rates(self):
        self.df_all = self.data_processing.data.copy().sort_values('idate')
        df_train, df_valid, df_test = data_split_with_date(self.df_all, self.split_rates)
        self.train_len, self.valid_len, self.test_len = len(df_train), len(df_valid), len(df_test)
        return df_train, df_valid, df_test

    def _data_arrange(self):
        """
        arrange data
        """
        self.df_all.drop('cited', axis=1, inplace=True)
        y = self.df_all.xi
        self.patnums_idates = self.df_all[['patnum', 'idate']]
        df_x = self.df_all.drop(['xi', 'patnum', 'idate'], axis=1)
        self.df_all = pd.concat([df_x, y, self.patnums_idates], axis=1)
        self.data_features = df_x.columns
        self.x_matrix = df_x.values
        self.y_vector = y.values

    def data_transformation(self, method=None):
        """
        Fit the scalar with training data, and use it to transform valid & test data
        @param method: str, the method to transform the data

        Todo. not completed
        """
        from sklearn import preprocessing
        # Todo
        self.x_scalar = preprocessing.minmax_scale(self.x_matrix[:self.train_len])
        self.y_scalar = preprocessing.minmax_scale(self.y_vector[:self.train_len])

    def get_train_valid_data_x(self):
        train_x, train_y = self.x_matrix[:self.train_len], self.y_vector[:self.train_len].reshape(-1, 1)
        valid_x, valid_y = self.x_matrix[self.train_len: self.train_len + self.valid_len], \
            self.y_vector[self.train_len: self.train_len + self.valid_len].reshape(-1, 1)
        return train_x, train_y, valid_x, valid_y

    def get_train_valid_data_text(self, method_name, scalar_y=LogScalar()):
        df_text = self.data_processing.text_data.copy()
        df_xi = self.df_all[['xi', 'patnum', 'idate']].copy()
        df_all = pd.merge(df_text, df_xi, on='patnum', how='left').sort_values('idate').drop(['idate', 'patnum'],
                                                                                             axis=1)
        xi_values = df_all.xi.values
        df_all['method_a'] = df_all.apply(text_data_method_a_apply, axis=1)
        df_all['method_b'] = df_all['brf_sum_text#text'].map(lambda x: ' ' if isinstance(x, float) else x[:500])
        data_text = df_all[method_name].to_list()
        text_train, text_valid = data_text[:self.train_len], data_text[
                                                             self.train_len:self.train_len + self.valid_len]
        train_y = scalar_y.transform(xi_values[:self.train_len])
        valid_y = scalar_y.transform(xi_values[self.train_len:self.train_len + self.valid_len])
        return text_train, text_valid, train_y, valid_y

    # 4. Fit models - only x variables
    def model_fit_x_variables(self):
        train_x, train_y, valid_x, valid_y = self.get_train_valid_data_x()
        scalar_names = ['none', 'minmax', 'standard', 'power']
        reg_names = ['rf', 'lasso', 'nn_1', 'nn_3']

        for scalar_name in scalar_names:
            if scalar_name == 'none':
                scalar_x = None
                train_x_scale, valid_x_scale = train_x, valid_x
            else:
                if scalar_name == 'minmax':
                    scalar_x = preprocessing.MinMaxScaler()
                elif scalar_name == 'standard':
                    scalar_x = preprocessing.StandardScaler()
                else:
                    scalar_x = preprocessing.QuantileTransformer(output_distribution='normal')
                train_x_scale = scalar_x.fit_transform(train_x)
                valid_x_scale = scalar_x.transform(valid_x)

            scalar_y = LogScalar()
            train_y_scale, valid_y_scale = scalar_y.fit_transform(train_y), scalar_y.transform(valid_y)
            self.search_regressors(train_x_scale, train_y_scale, valid_x_scale, valid_y_scale, scalar_y,
                                   reg_names=reg_names, scalar_set=(scalar_name, scalar_x))

    # 5. Fit models - only text variables
    def model_fit_texts(self):
        """
        Module Two: text data
        For text data, try two different ways
         a) Take the first claims text (e.g., claim #1, it is numbered) and concatenate it with "Abstract".
            Cut it at 500 words total. Put claims #1 front.
         b) just use the first 500 words in brf_sum_txt
        """
        processor_names = [None, 'lda', 'pca', 'isomap', 'mds', 'tsne']
        reg_names = ['rf', 'lasso', 'nn_1', 'nn_3']
        scalar_y = LogScalar()

        for method_name in ['method_a', 'method_b']:
            text_train, text_valid, train_y, valid_y = self.get_train_valid_data_text(method_name, scalar_y)

            for vocab_size in [100, 300, 500]:
                count_vectorizer = CountVectorizer(max_df=0.95, min_df=3, stop_words='english', max_features=vocab_size)
                tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english', max_features=vocab_size)

                for vectorizer in [count_vectorizer, tfidf_vectorizer]:
                    train_x = vectorizer.fit_transform(text_train).toarray()
                    valid_x = vectorizer.transform(text_valid)

                    for processor_name in processor_names:
                        # todo: search for n_components
                        n_components = 10

                        if processor_name is not None:
                            processor = get_processor(processor_name, n_components)
                            train_x = processor.fit_transform(train_x)
                            valid_x = processor.transform(valid_x)
                        else:
                            processor_name = 'none'

                        vectorizer_name = 'none' if vectorizer is None else str(vectorizer.__class__).split('.')[-1][:-2]
                        text_operations_name = '_'.join([method_name, str(vocab_size), vectorizer_name,
                                                         str(n_components), processor_name])
                        self.search_regressors(train_x, train_y, valid_x, valid_y, scalar_y, reg_names=reg_names,
                                               text_operations_name=text_operations_name)

    # 6. Fit models - both x variables and text variables
    def model_fit_all(self):
        # Module Three:
        # according to previous analysis:
        scalar_name = 'minmax'
        scalar_x, scalar_y = preprocessing.MinMaxScaler(), LogScalar()
        vocab_size = 500
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english', max_features=vocab_size)
        processor_name, processor = 'none', None
        n_components = 10  # TODO: try more
        method_names = ['method_a', 'method_b']
        reg_names = ['rf', 'nn_3']

        # get x variables
        train_x, train_y, valid_x, valid_y = self.get_train_valid_data_x()
        train_x_scale, valid_x_scale = scalar_x.fit_transform(train_x), scalar_x.transform(valid_x)
        train_y_scale, valid_y_scale = scalar_y.fit_transform(train_y), scalar_y.transform(valid_y)

        for method_name in method_names:
            # get text variables
            text_train, text_valid, train_y_scale_text, valid_y_scale_text = \
                self.get_train_valid_data_text(method_name, scalar_y=LogScalar())
            train_x_text = vectorizer.fit_transform(text_train).toarray()
            valid_x_text = vectorizer.transform(text_valid).toarray()
            vectorizer_name = 'none' if vectorizer is None else str(vectorizer.__class__).split('.')[-1][:-2]
            text_operations_name = '_'.join([method_name, str(vocab_size), vectorizer_name,
                                             str(n_components), processor_name, scalar_name])

            # merge two kinds of variables
            train_x_all = np.concatenate([train_x_scale, train_x_text], axis=1)
            valid_x_all = np.concatenate([valid_x_scale, valid_x_text], axis=1)
            self.search_regressors(train_x_all, train_y_scale, valid_x_all, valid_y_scale, scalar_y, reg_names=reg_names,
                                   text_operations_name=text_operations_name)

    def search_regressors(self, train_x, train_y, valid_x, valid_y, scalar_y, reg_names=('rf', 'lasso', 'nn_1', 'nn_3'),
                          scalar_set=None, text_operations_name=None):
        precessor_name = ''
        if scalar_set is not None:
            scalar_name, scalar_x = scalar_set
            precessor_name += scalar_name
        if text_operations_name is not None:
            precessor_name += text_operations_name

        for reg_name in reg_names:

            if reg_name == 'rf':
                hyperparameter_settings = [(num_trees, max_depth)
                                           for num_trees in [200]
                                           for max_depth in [10]]
                for hyperparameter_setting in hyperparameter_settings:
                    if check_dict_text([precessor_name, reg_name, str(hyperparameter_setting)]):
                        continue
                    reg = RandomForestRegressor(n_estimators=hyperparameter_setting[0],
                                                max_depth=hyperparameter_setting[1])
                    reg, mae_scale, mae = calculate_mae(reg, train_x, train_y, valid_y, valid_x, scalar_y)
                    self.dict_performance = add_performance(precessor_name, reg_name, hyperparameter_setting,
                                                            mae_scale, mae, self.dict_performance)

            elif reg_name == 'lasso':
                hyperparameter_settings = [alpha for alpha in [0.001, 0.01, 0.1, 1, 10, 100]]
                for hyperparameter_setting in hyperparameter_settings:
                    if check_dict_text([precessor_name, reg_name, str(hyperparameter_setting)]):
                        continue
                    reg = Lasso(alpha=hyperparameter_setting)
                    reg, mae_scale, mae = calculate_mae(reg, train_x, train_y, valid_y, valid_x, scalar_y)
                    self.dict_performance = add_performance(precessor_name, reg_name, hyperparameter_setting,
                                                            mae_scale, mae, self.dict_performance)

            else:
                hyperparameter_settings = [(n, activation, optimizer, epoch)
                                           for n in [10]
                                           for activation in ['elu', 'selu']  # 'relu',
                                           for optimizer in ['rmsprop', 'nadam']  # 'sgd', 'adam', 'adagrad',
                                           for epoch in [50]]  # 30,
                for hyperparameter_setting in hyperparameter_settings:
                    if check_dict_text([precessor_name, reg_name, str(hyperparameter_setting)]):
                        continue
                    reg = Sequential()
                    if reg_name == 'nn_3':
                        reg.add(Dense(hyperparameter_setting[0], activation=hyperparameter_setting[1]))
                        reg.add(Dense(hyperparameter_setting[0], activation=hyperparameter_setting[1]))
                    reg.add(Dense(1, activation=hyperparameter_setting[1], input_shape=(train_x.shape[-1],)))
                    reg.compile(optimizer=hyperparameter_setting[2], loss='mean_absolute_error',
                                metrics=['mean_absolute_error'])
                    history = reg.fit(train_x, train_y,
                                      epochs=hyperparameter_setting[3], batch_size=256, validation_split=0.3)
                    test_loss, mae_scale = reg.evaluate(valid_x, valid_y)
                    if scalar_y is None:
                        mae = mae_scale
                    else:
                        predicted = scalar_y.inverse_transform(reg.predict(valid_x))
                        mae = mean_absolute_error(scalar_y.inverse_transform(valid_y), predicted)
                    self.dict_performance = add_performance(precessor_name, reg_name, hyperparameter_setting,
                                                            mae_scale, mae, self.dict_performance)
                    # plot loss during training
                    # plot_nn(history, hyperparameter_setting)


if __name__ == '__main':

    test = False
    if test:
        with open('data/model.obj', 'rb') as file:
            self = pickle.load(file)
    else:
        model = Model(split_rates=(0.7, 0.15, 0.15))
        self = model

        # file_model = open('data/model.obj', 'wb')
        # pickle.dump(model, file_model)
        # file_model.close()
        del model

    # self.model_fit_x_variables()
    self.model_fit_texts()
    # with open('data/dict_performance.obj', 'rb') as file:
    #     dict_performance_read = pickle.load(file)



