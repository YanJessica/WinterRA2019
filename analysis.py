from Model import *


def analysis_x(filename='data/text_performance-x.txt'):
    # only x variables
    # read file and construct data frame for MAEs
    df_mae_x = analysis_construct_mae_df(filename, ['scalar_name', 'reg_name', 'reg_paras'])

    # get group-by aggregated data. Check each df manually
    df = df_mae_x.drop('mae_scale', axis=1).copy()
    agg_funcs = ['min', 'mean', 'median', 'max']
    df_reg_name = df.groupby('reg_name').agg(agg_funcs)
    df_scalar = df.groupby('scalar_name').agg(agg_funcs)

    # construct data frame only for NNs
    df_nn = df[df.reg_name.isin(['nn-1', 'nn-3'])].copy()
    df_nn['num_layer'] = df_nn.reg_name.map(lambda x: int(x[-1]))
    nn_para_names = ['n_neurons', 'activation', 'optimizer', 'epochs']
    df_nn = clean_column_reg_paras(df_nn, nn_para_names)
    # check the group-by aggregated values for each parameter
    to_check = nn_para_names+['scalar_name', 'num_layer']
    check_groupby_agg(df_nn, to_check, agg_funcs)
    # plot mae distributions for each parameter
    df_nn_1 = df_nn.copy()
    df_nn_1.mae[df_nn_1.mae > 100] = 100
    plot_mae_dist_violin(df_nn_1, to_check)
    # plot after removing large values
    df_nn_2 = df_nn_1.replace(100, np.NaN).dropna()
    plot_mae_dist_violin(df_nn_2, to_check)
    # NOTE: n_neurons=10, activation='selu'/'elu', optimizer='nadam'/'rmsprop',
    #       epochs=50, num_layer=3, scalar_name='minmax'
    # TODO: try larger num_layer

    # construct data frame only for RFs
    df_rf = df[df.reg_name == 'rf'].copy()
    rf_para_names = ['n_trees', 'max_depth']
    df_rf = clean_column_reg_paras(df_rf, rf_para_names)
    to_check = rf_para_names+['scalar_name']
    check_groupby_agg(df_rf, to_check)
    plot_mae_dist_violin(df_rf, to_check)
    rf_paras_dict = {'n_tree': 200, 'max_depth': 10, 'scalar_name': 'minmax'}
    # NOTE: n_tree=200, max_depth=10, scalar_name='minmax'/'standard'/none
    # TODO: try larger max_depth

    # construct data frame only for RFs
    df_lasso = df[df.reg_name == 'lasso'].copy()
    lasso_para_names = ['alpha']
    df_lasso[lasso_para_names[0]] = df_lasso.reg_paras
    to_check = lasso_para_names+['scalar_name']
    check_groupby_agg(df_lasso, to_check)
    df_lasso.groupby(to_check).agg(agg_funcs)
    df_lasso_1 = df_lasso.copy().replace('power', np.NaN).dropna()
    check_groupby_agg(df_lasso_1, to_check)
    plot_mae_dist_violin(df_lasso_1, to_check)
    # NOTE: alpha=0.01, scalar_name='minmax'


def explain_x():
    # Explainability
    self = Model(split_rates=(0.7, 0.15, 0.15))
    train_x, train_y, valid_x, valid_y = self.get_train_valid_data_x()
    scalar_name, scalar_x, scalar_y = 'minmax', preprocessing.MinMaxScaler(), LogScalar()
    train_x_scale, valid_x_scale = scalar_x.fit_transform(train_x), scalar_x.transform(valid_x)
    train_y_scale, valid_y_scale = scalar_y.fit_transform(train_y), scalar_y.transform(valid_y)
    features = np.array(self.data_features)

    # 1. RF variable importance
    reg_name = 'rf'
    reg_rf = RandomForestRegressor(n_estimators=200, max_depth=10)
    reg_rf, mae_scale_rf, mae_rf = calculate_mae(reg_rf, train_x_scale, train_y_scale,
                                                 valid_y_scale, valid_x_scale, scalar_y)
    print("Reg name: %s, mae: %.2f" % (reg_name, mae_rf))
    # Calculate feature importance
    importance = reg_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in reg_rf.estimators_], axis=0)
    indices = np.argsort(importance)[::-1]
    print("Feature ranking:")  # Print the feature ranking
    for i, idx in enumerate(indices):
        print("%-3d. feature %-60s %.5f" % (i + 1, features[idx], importance[idx]))
    # Plot the feature importance of the forest
    bar = 0.01
    indices_remain = indices[: sum(importance >= bar)]
    features_remain_rf = features[indices_remain]
    plt.figure()
    plt.bar(range(len(indices_remain)), importance[indices_remain], color="r", yerr=std[indices_remain], align="center")
    plt.title("Feature importances")
    plt.xticks(range(len(indices_remain)), features[indices_remain])  # , rotation='vertical')
    plt.xticks(rotation=45)
    plt.show()

    # 2. LASSO parameter coefficients
    reg_name = 'lasso'
    reg_lasso = Lasso(alpha=0.01)
    reg_lasso, mae_scale_lasso, mae_lasso = calculate_mae(reg_lasso, train_x_scale, train_y_scale,
                                                          valid_y_scale, valid_x_scale, scalar_y)
    # calculate coefficients
    coefficients = reg_lasso.coef_
    indices_not_zero = np.arange(0, len(coefficients), 1)[coefficients!=0]
    features_remain_lasso = features[indices_not_zero]
    coefficients_remain = coefficients[indices_not_zero]
    # print coefficients
    indices = np.argsort(coefficients_remain)[::-1]
    print("Coefficients remain:")  # Print the feature ranking
    for i, idx in enumerate(indices):
        print("%-3d. feature %-60s %.5f" % (i + 1, features_remain_lasso[idx], coefficients_remain[idx]))
    # plot coefficients
    plt.figure()
    plt.bar(range(len(features_remain_lasso)), coefficients_remain[indices], color="r", align="center")
    plt.title("Feature coefficients by Lasso")
    plt.xticks(range(len(indices)), np.array(features_remain_lasso))  # , rotation='vertical')
    plt.xticks(rotation=45)
    plt.show()
    # alpha plot
    lasso_alpha_plot(train_x_scale, train_y_scale)
    # NOTE: when alpha=0.00001, Tsm_avg has the largest coefficient (positive)
    #       and Tsm_sum has the smallest coefficient (negative)

    print(set(features_remain_lasso) & set(features_remain_rf))
    print(set(features_remain_rf) - set(features_remain_lasso))
    print(set(features_remain_lasso) - set(features_remain_rf))
    # NOTE: both rf and lasso have: {'permno', 'Tsm_avg', 'Tcw_sum', 'fNpats_sum'}
    #       only rf has:     {'Tsm_sum', 'num_previous_patent', 'Tcw_avg', 'fNpats_avg'}
    #       only lasso has:  {'application#series_code_8', 'class_xi_more_than_20', 'fNpats_length'}


def analysis_text(filename='data/text_performance-text.txt'):
    idx_col_names_text = ['method_name', 'n_vocab', 'vectorizer_name',
                          'n_component', 'processor_name', 'reg_name', 'reg_paras']
    df_mae_text = analysis_construct_mae_df(filename, idx_col_names_text)
    df = df_mae_text.drop('mae_scale', axis=1).copy()
    # NOTE: optimal: method_name='method-b', n_vocab=500,vectorizer_name='TfidfVectorizer', processor_name=none,
    #       reg_name=nn-3, reg_paras=(10,'selu','nadam',50)  --> mae=13.9104
    # TODO: try more complicated NN. try more n_component.

    agg_funcs = ['min', 'mean', 'median', 'max']
    for col_name in idx_col_names_text[:-1]:
        print(df.groupby(col_name).agg(agg_funcs))
    # NOTE: min MAEs: nn-3: 13.9104; rf: 13.9280; nn-1: 14.0619; lasso  14.1377;
    plot_mae_dist_violin(df, idx_col_names_text[:-1])
    print(df.groupby(['method_name', 'n_vocab']).agg(agg_funcs))

    sns.violinplot(x='n_vocab', y='mae', data=df, hue='method_name')
    plt.show()

    sns.violinplot(x='method_name', y='mae', data=df, hue='vectorizer_name')
    plt.show()

    sns.violinplot(x='processor_name', y='mae', data=df, hue='method_name')
    plt.show()

    sns.violinplot(x='reg_name', y='mae', data=df, hue='method_name')
    plt.show()


