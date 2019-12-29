import pickle
from static_funcs import *
from datetime import datetime
import datetime


class DataPreProcessing:

    def __init__(self, filename='data/patents_data_94_97_combined.pkl'):

        # open data
        with open(filename, 'rb') as f:
            self.data_all = pickle.load(f)
        self.data = self.data_all
        self.text_data = None
        self.lists_col_names, self.text_col_names, self.cate_col_names, self.drop_col_names = [], [], [], []

        self._feature_engineer()     # Feature engineering
        self.drop_features()         # Drop features
        self._fill_missing_values()  # Fill missing values

    def _feature_engineer(self):
        # Feature Engineering
        df = self.data.copy()

        # 1. length of the list
        trans_lst = ['location#state_fips', 'location#county_fips', 'cpc_current#section_id',
                     'cpc_current#group_id', 'cpc_current#subgroup_id', 'icpr#section', 'wipo_field#sector_title',
                     'wipo_field#field_title', 'cpc_current#subsection_id']
        for col_name in trans_lst:
            df["length_"+col_name] = df[col_name].map(lambda x: 0 if isinstance(x, float) else len(x))

        # 2. Number of days between issue date and file date
        for col_name in ['fdate', 'idate']:
            df[col_name] = df[col_name].map(lambda x: pd.Timestamp(x))
        df['num_days_fdate_idate'] = (df.fdate-df.idate).map(lambda x: np.abs(x.days))

        # 3. fNpats: sum value, mean value, length
        for col_name in ['fNpats', 'Tcw', 'Tsm']:
            df[col_name+'_drop_nan'] = df[col_name].map(lambda x: x if None not in x else [i for i in x if i is not None])
            df[col_name+'_sum'] = df[col_name+'_drop_nan'].map(lambda x: np.sum(x))
            df[col_name+'_avg'] = df[col_name+'_drop_nan'].map(lambda x: np.mean(x))
            df = df.drop(col_name+'_drop_nan', axis=1)
        df['fNpats_length'] = df['fNpats'].map(lambda x: len(x))

        # 4. uspatentcitation#citation_id_date_category:
        #    the avg num of month between the selected patent and its citing ones.
        df['uspatentcitation#citation_id_date_category_avg_month'] = df.apply(lambda row: func_avg_month(row), axis=1)

        # 5. nciting: number of citing
        df['nciting'] = df.citing.map(lambda x: len(x))
        df['ncited_all'] = df.ncites

        # 6. Number of patents issued by this permno before idate (permno)
        df1 = df.copy().sort_values('idate').groupby('permno').apply(permno_group_func)
        df = pd.merge(df, df1, on='patnum')

        # 7. transform a categorical variable - application#series_code
        df['application#series_code_8'] = df['application#series_code'].map(lambda x: 1 if int(x) == 8 else 0)
        df['application#series_code_7'] = df['application#series_code'].map(lambda x: 1 if int(x) == 7 else 0)
        df['application#series_code_others'] = df['application#series_code'].map(
            lambda x: 1 if int(x) != 7 and int(x) != 8 else 0)

        self.data = df

    def drop_features(self, drop_list=None):
        # drop variables
        df = self.data
        if drop_list is None:
            drop_list = []

        # 1. all list variables
        lists_col_names = []
        for col_name in df.columns:
            if isinstance(df[col_name][0], list):
                lists_col_names.append(col_name)
        self.lists_col_names = list(set(lists_col_names) - {'cited'})

        # 2. all text variables
        self.text_col_names = ['brf_sum_text#text', 'claim#text', 'patent#abstract', 'patent#title',
                               'cpc_subsection#title']
        self.text_data = df[['patnum']+self.text_col_names]
        # data_text.to_pickle("./data/text_data.pkl")

        # 3. categorical variables to be transformed before spliting
        self.cate_col_names = ['class', 'subclass']

        # 4. to be dropped
        self.drop_col_names = ['patent_id', 'patent#date', 'patent#kind', 'application#date', 'patent#country', 'pdate',
                               'patent#withdrawn', 'application#country', 'patent#type', 'ncites', 'fdate',
                               'application#series_code']

        # drop variables
        df = df.drop(self.lists_col_names+self.text_col_names+self.drop_col_names+drop_list, axis=1)
        self.data = df

    def _fill_missing_values(self):
        """
        Only two features have missing values: 'figures#num_figures', 'figures#num_sheets'.
        To fill those missing values, I want to use the average value of samples that have the same class with the
        target sample's class. To avoid using future data, those missing values should be filled before data splitting.
        """
        # df = self.data
        # for col_name in df.columns:
        #     num_na = sum(df[col_name].isnull())
        #     if num_na != 0:
        #         print(col_name, "     ", num_na)
        # check_col_names = ['figures#num_figures', 'figures#num_sheets']
        # cate_name = check_col_names[1]
        # for cate_name in check_col_names:
        #     df1 = df[['xi', cate_name]].copy()
        #     df1_value_counts = df1[cate_name].value_counts()
        #     df1_agg = df1.groupby(cate_name).agg(['count', 'min', 'max', 'mean'])
        #     df1_agg[('xi', 'mean')].plot()
        #     plt.show()
        # df[['xi'] + check_col_names].corr()
        pass


if __name__ == '__main__':
    data_processing = DataPreProcessing()
    df = data_processing.data.copy()

    file_data_pre_processing = open('data/data_process.obj', 'wb')
    pickle.dump(data_processing, file_data_pre_processing)
    file_data_pre_processing.close()


# note: variables to be dealt with before data splitting:
#  1. 'ncited_all', 'class', 'subclass': only count ones before the splitting date.
#  2. 'figures#num_figures', 'figures#num_sheets': fill missing values based on the splitting date.


# # Check remaining columns
# col_name_left = set(df.columns) - set(data_processing.lists_col_names+data_processing.text_col_names+
#                                       data_processing.cate_col_names+data_processing.drop_col_names)
# string_cols = []
# for col_name in col_name_left:
#     print(df[col_name].dtype, "          ", col_name)
#     if df[col_name].dtype == 'object':
#         string_cols.append(col_name)

# # Does the categorical variable matter?
# df = data_processing.data_all.copy().sort_values('idate')
# cate_col_names = ['class', 'subclass']
# cate_name = cate_col_names[1]
# for cate_name in cate_col_names:
#     df1_value_counts, df1_plot = plot_categorical_variable(df, cate_name)




