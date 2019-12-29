import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.linear_model import Lasso


def func_avg_month(row):
    if isinstance(row['uspatentcitation#citation_id_date_category'], float):
        return 0
    dates = [x[1] for x in row['uspatentcitation#citation_id_date_category']
             if 1700 <= int(x[1][:4]) <= row.idate.year and '-' in x[1]]
    if len(dates) == 0:
        return 0
    if dates[0][-2:] != '01':
        dates = [x[:-2] + '01' for x in dates]
    month_lst = [round((row.idate - pd.Timestamp(x)).days / 30, 0) for x in dates]
    return np.round(np.mean(month_lst), 0)


def permno_group_func(group):
    group['num_previous_patent'] = range(len(group))
    return group[['patnum', 'num_previous_patent']]


def plot_categorical_variable(df, col_name, plot=False):
    df1 = df[['xi', col_name]]
    df1_value_counts = df1[col_name].value_counts()
    df1_agg = df1.groupby(col_name).agg(['count', 'min', 'max', 'mean'])
    df1_plot = df1_agg[('xi',  'mean')].copy()
    df1_plot = df1_plot.sort_values()
    if plot:
        df1_plot.plot()
        plt.show()
    return df1_value_counts, df1_plot


def fill_missing_values(df1, col_name):
    avg_value = df1[col_name].mean()
    df1_mean = np.round(df1.groupby('class').mean(), 0).reset_index().drop('patnum', axis=1)
    df1 = pd.merge(df1, df1_mean, how='left', on='class')
    df1[col_name] = df1.apply(lambda row: row[col_name+'_x'] if not np.isnan(row[col_name+'_x']) else
                              row[col_name+'_y'] if not np.isnan(row[col_name+'_x']) else avg_value, axis=1)
    return df1


def data_split_with_date(df, split_rates):
    train_valid_date = df.idate[int(len(df) * split_rates[0])]
    valid_test_date = df.idate[len(df) - int(len(df) * split_rates[-1])]
    df_train = df[df.idate <= train_valid_date].copy()
    if split_rates[1] == 0:
        df_valid = df[df.idate > train_valid_date].copy()
        return df_train, None, df_valid
    else:
        df_valid = df[(train_valid_date < df.idate) & (df.idate <= valid_test_date)].copy()
        df_test = df[df.idate > valid_test_date].copy()
        return df_train, df_valid, df_test


def data_missing_fill(df_train, df_valid, df_test, col_name):
    """
    Fill missing values ('figures#num_figures', 'figures#num_sheets') based on class.
    If the class has no non-missing value, then fill with the mean value of all data.
    @param df_all: the data frame
    @param split_rates: train-valid-test
    @param col_name: the column to be filled
    @return:
    """
    # train data fill
    df_train_fill = fill_missing_values(df_train[['patnum', 'class', col_name]].copy(), col_name)
    df_train = pd.merge(df_train.drop(col_name, axis=1),
                        df_train_fill[['patnum', col_name]], on='patnum')
    # valid data fill
    df_valid_fill = fill_missing_values(pd.concat([df_train, df_valid])
                                        [['patnum', 'class', col_name]].copy(), col_name)
    df_valid = pd.merge(df_valid.drop(col_name, axis=1), df_valid_fill[['patnum', col_name]],
                        on='patnum', how='left')
    # test data fill
    df_test_fill = fill_missing_values(pd.concat([df_train, df_valid, df_test])
                                       [['patnum', 'class', col_name]].copy(), col_name)
    df_test = pd.merge(df_test.drop(col_name, axis=1), df_test_fill[['patnum', col_name]],
                       on='patnum', how='left')
    return df_train, df_valid, df_test


def class_subclass_processing(df_train, df_valid, df_test):
    """
    to deal with the variable class and the variable subclass
    replace them with dummy variables
    """
    df_class_value_counts, df_class_plot = plot_categorical_variable(df_train, 'class')
    df_subclass_value_counts, df_subclass_plot = plot_categorical_variable(df_train, 'subclass')
    classes_xi_more_than_20 = df_class_plot.index[df_class_plot > 20].tolist()
    subclasses_xi_more_than_50 = df_subclass_plot.index[df_subclass_plot > 50].tolist()
    df = pd.concat([df_train, df_valid, df_test])
    df['class_xi_more_than_20'] = df['class'].map(lambda x: 1 if x in classes_xi_more_than_20 else 0)
    df['subclass_xi_more_than_50'] = df['subclass'].map(lambda x: 1 if x in subclasses_xi_more_than_50 else 0)
    df = df.drop(['class', 'subclass'], axis=1)
    return df


def n_cited_calculation(df_all, df_train, df_valid, df_test):
    """
    calculate the number of cited before the observation date.
    cited patents - only count ones before the splitting date. Note: Patents issued later have larger patnums
    """
    df_patnum_idate = df_all[['patnum', 'idate']].copy().sort_values(['patnum'])
    max_cited_patnum_train = max(df_patnum_idate[df_patnum_idate.idate <= max(df_train.idate)].patnum)
    max_cited_patnum_valid = max(df_patnum_idate[df_patnum_idate.idate <= max(df_valid.idate)].patnum)
    max_cited_patnum_test = df_patnum_idate.patnum.tolist()[-1]
    df_train['ncited'] = df_train.cited.map(lambda lst: len([x for x in lst if int(x) <= max_cited_patnum_train]))
    df_valid['ncited'] = df_valid.cited.map(lambda lst: len([x for x in lst if int(x) <= max_cited_patnum_valid]))
    df_test['ncited'] = df_test.cited.map(lambda lst: len([x for x in lst if int(x) <= max_cited_patnum_test]))
    return df_train, df_valid, df_test


def save_dict(dict_performance, file_name='data/dict_performance.obj'):
    file_d = open(file_name, 'wb')
    pickle.dump(dict_performance, file_d)
    file_d.close()


def check_dict_text(search, file_name='data/text_performance.txt'):
    with open(file_name, 'r') as f:
        if ' '.join(search) in f.read():
            return True
        else:
            return False


def save_dict_text(tuple_towirte, file_name='data/text_performance.txt'):
    file_text = open(file_name, 'a')
    file_text.write('_'.join(tuple_towirte[:3])+' '+tuple_towirte[3]+' '+tuple_towirte[4]+'\n')
    file_text.close()


def add_performance(precessor_name, reg_name, hyperparameter_setting, mae_scale, mae,
                    dict_performance):
    print("Processor name: %s, Reg name: %s, paras: %s, mae_scale: %.2f, mae: %.2f" %
          (precessor_name, reg_name, str(hyperparameter_setting), mae_scale, mae))
    dict_performance['processor_name'].append(precessor_name)
    dict_performance['reg_name'].append(reg_name)
    dict_performance['hyperparameter_setting'].append(hyperparameter_setting)
    dict_performance['mae_scale'].append(mae_scale)
    dict_performance['mae'].append(mae)
    # save_dict(dict_performance)
    save_dict_text((precessor_name, reg_name, str(hyperparameter_setting), str(round(mae_scale, 4)), str(round(mae, 4))))
    return dict_performance


def calculate_mae(reg, train_x_scale, train_y_scale, valid_y_scale, valid_x_scale, scalar_y=None):
    reg.fit(train_x_scale, train_y_scale.ravel())
    mae_scale = mean_absolute_error(valid_y_scale.ravel(), reg.predict(valid_x_scale))
    if scalar_y is not None:
        mae = mean_absolute_error(scalar_y.inverse_transform(valid_y_scale),
                                  scalar_y.inverse_transform(reg.predict(valid_x_scale).reshape(-1, 1)))
    else:
        mae = mae_scale
    return reg, mae_scale, mae


def text_data_method_a_apply(row):
    a, b = row['claim#text'], row['patent#abstract']
    if a is None or isinstance(a, float) or len(a) == 0:
        res = ' ' if b is None or isinstance(b, float) else b  # or np.isnan(b)
    else:
        res = a[0] if b is None or isinstance(b, float) else a[0]+' '+b
    return res[:500]


class LogScalar:
    def transform(self, data):
        return np.log(data)

    def fit_transform(self, data):
        return np.log(data)

    def inverse_transform(self, data):
        if sum(data.ravel()>100) > 0:
            return np.full(data.shape, np.inf)
        return np.exp(data)


def get_processor(processor_name, n_components):
    if processor_name == 'lda':
        processor = LatentDirichletAllocation(n_components=n_components, learning_method='online')
    elif processor_name == 'pca':
        processor = PCA(n_components=n_components)
    elif processor_name == 'isomap':
        processor = Isomap(n_components=n_components)  # todo: n_neighbors
    elif processor_name == 'mds':
        processor = MDS(n_components=n_components, dissimilarity='precomputed')
    else:
        processor = TSNE(n_components=n_components)  # todo: perplexity, learning_rate
    return processor


def plot_nn(history, hyperparameter_setting):
    plt.title('Loss: MAE\n' + str(hyperparameter_setting))
    plt.plot(history.history['loss'], label='train-train')
    plt.plot(history.history['val_loss'], label='train-valid')
    plt.legend()
    plt.show()


def plot_mae_dist_violin(df, to_check, num_col=3):
    plt.figure()
    fig, axes = plt.subplots(int(np.ceil(len(to_check) / num_col)), num_col)
    for i, col_name in enumerate(to_check):
        ax = axes[i % num_col] if len(axes.shape)==1 else axes[int(i / num_col)][i % num_col]
        sns.violinplot(x=col_name, y='mae', data=df, ax=ax)
        if max(df[col_name].map(lambda x: len(str(x)))) > 3:
            # print(col_name)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    plt.show()


def clean_column_reg_paras(df, para_names):
    df_paras = df.reg_paras.map(lambda x: x[1:-1]).str.split(',', expand=True)
    df_paras.columns = para_names
    df = pd.concat([df, df_paras], axis=1)
    return df


def check_groupby_agg(df, to_check, agg_funcs=None):
    if agg_funcs is None:
        agg_funcs = ['min', 'mean', 'median', 'max']
    for col_name in to_check:
        print(df[[col_name, 'mae']].groupby(col_name).agg(agg_funcs))


def lasso_alpha_plot(train_x_scale, train_y_scale, n_alpha=50):
    alphas = np.logspace(-5, 2, n_alpha)
    res = []
    for a in alphas:
        lasso_1 = Lasso(alpha=a, fit_intercept=False)
        lasso_1.fit(train_x_scale, train_y_scale)
        res.append(lasso_1.coef_)
    ax = plt.gca()
    ax.plot(alphas, res)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()


def analysis_read_performance_file(filename):
    lst_paras, lst_mae_scale, lst_mae = [], [], []
    with open(filename, 'r') as f:
        for line in f.read().replace('method_', 'method-').replace('nn_', 'nn-').split('\n'):
            if len(line) < 5:
                continue
            lst_line = line.split(' ')
            lst_paras.append(''.join(lst_line[:-2]))
            lst_mae_scale.append(float(lst_line[-2]))
            lst_mae.append(float(lst_line[-1]))
    return lst_paras, lst_mae_scale, lst_mae


def analysis_construct_mae_df(filename, index_col_names):
    # read file
    lst_paras, lst_mae_scale, lst_mae = analysis_read_performance_file(filename)
    df_index = pd.Series(lst_paras).str.split('_', expand=True)
    df_index.columns = index_col_names
    df_mae = pd.concat([df_index, pd.DataFrame({'mae_scale': lst_mae_scale, 'mae': lst_mae})], axis=1)
    df_mae = df_mae.sort_values('mae')
    return df_mae


