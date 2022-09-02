import pyarrow.parquet as pq
from m3gp.M3GP import M3GP
from sklearn.model_selection import train_test_split
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
pca = PCA()

def pca_decomposition(Tr_X,Tr_Y, Te_X, Te_Y):
    X_train_pca = pca.fit_transform(Tr_X)
    X_test_pca = pca.transform(Te_X)
    fitted_model_X= pca.fit(Tr_X)
    X = pd.DataFrame(np.vstack((X_train_pca, X_test_pca)))
    Y = pd.concat([Tr_Y, Te_Y], ignore_index=True)
    return pd.concat([X, Y], axis=1), fitted_model_X


def load_data(train_path, val_path,test_path):
    ds_train = pq.ParquetDataset(train_path)
    ds_validation = pq.ParquetDataset(val_path)
    ds_test = pq.ParquetDataset(test_path)

    train_samples = ds_train.read().to_pandas()
    validation_samples = ds_validation.read().to_pandas()
    test_samples = ds_test.read().to_pandas()
    return pd.concat([train_samples, validation_samples, test_samples])

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


#merged["ti"] = pd.to_numeric(weather.Temp, errors='coerce')
def preprocess(merged):
    merged = merged.reindex(sorted(merged.columns), axis=1)

    #merged = merged.drop_duplicates(['class', 'tile_id'])
    #drop some features
    merged = merged.drop(['class_pct', 'class_label', 'date', 'tile_id'], axis=1)
    #encode region
    encoder = LabelEncoder()
    merged['area'] = encoder.fit_transform(merged['area'])


    #compute some additional features like interquantile ranges, stds, means, ndvi, seasonal changes
    merged['clm_accum_monthly_precipitation_mean'] = merged[[
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.01.01...01.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.02.01...02.28',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.03.01...03.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.04.01...04.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.05.01...05.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.06.01...06.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.07.01...07.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.08.01...08.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.09.01...09.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.10.01...10.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.11.01...11.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_accum_monthly_precipitation_std'] = merged[[
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.01.01...01.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.02.01...02.28',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.03.01...03.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.04.01...04.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.05.01...05.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.06.01...06.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.07.01...07.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.08.01...08.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.09.01...09.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.10.01...10.31',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.11.01...11.30',
         'clm_accum.precipitation_chelsa.montlhy_m_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.daytime.m02_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m02_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m02_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m03_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m03_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m03_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m04_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m04_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m04_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m05_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m05_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m05_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m06_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m06_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m06_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m07_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m07_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m07_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m08_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m08_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m08_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m09_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m09_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m09_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m10_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m10_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m10_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m11_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m11_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m11_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.daytime.m12_iqr_p95_p05'] = merged['clm_lst_mod11a2.daytime.m12_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.daytime.m12_p05_1km_s0..0cm_2000..2021']

    merged['clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_mean'] = \
        merged[['clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.01.01...01.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.02.01...02.28',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.03.01...03.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.04.01...04.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.05.01...05.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.06.01...06.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.07.01...07.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.08.01...08.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.09.01...09.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.10.01...10.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.11.01...11.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.12.01...12.31' ]].mean(axis=1)

    merged['clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_std'] = \
        merged[['clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.01.01...01.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.02.01...02.28',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.03.01...03.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.04.01...04.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.05.01...05.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.06.01...06.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.07.01...07.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.08.01...08.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.09.01...09.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.10.01...10.31',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.11.01...11.30',
             'clm_lst_mod11a2.daytime.trend_p50_1km_s0..0cm_.12.01...12.31' ]].std(axis=1)

    merged['clm_lst_mod11a2.daytime_p05_1km_s0..0cm_mean'] = \
        merged[['clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.01.01...01.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.02.01...02.28',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.03.01...03.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.04.01...04.30',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.05.01...05.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.06.01...06.30',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.07.01...07.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.08.01...08.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.09.01...09.30',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.10.01...10.31',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.11.01...11.30',
            'clm_lst_mod11a2.daytime_p05_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_lst_mod11a2.daytime_p05_1km_s0..0cm_std'] = \
        merged[['clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.01.01...01.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.02.01...02.28',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.03.01...03.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.04.01...04.30',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.05.01...05.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.06.01...06.30',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.07.01...07.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.08.01...08.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.09.01...09.30',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.10.01...10.31',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.11.01...11.30',
            'clm_lst_mod11a2.daytime_p50_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.daytime_p05_1km_s0..0cm_std'] = \
        merged[[
            'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.01.01...01.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.02.01...02.28',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.03.01...03.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.04.01...04.30',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.05.01...05.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.06.01...06.30',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.07.01...07.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.08.01...08.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.09.01...09.30',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.10.01...10.31',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.11.01...11.30',
             'clm_lst_mod11a2.daytime_p95_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.nighttime.m02_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m02_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m02_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m03_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m03_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m03_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m04_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m04_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m04_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m05_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m05_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m05_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m06_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m06_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m06_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m07_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m07_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m07_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m08_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m08_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m08_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m09_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m09_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m09_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m10_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m10_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m10_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m11_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m11_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m11_p05_1km_s0..0cm_2000..2021']
    merged['clm_lst_mod11a2.nighttime.m12_p95_p05'] = merged['clm_lst_mod11a2.nighttime.m12_p95_1km_s0..0cm_2000..2021'] - merged['clm_lst_mod11a2.nighttime.m12_p05_1km_s0..0cm_2000..2021']

    merged['clm_lst_mod11a2.nighttime.trend_p50_1km_s0_mean'] = \
        merged[['clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.01.01...01.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.02.01...02.28',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.03.01...03.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.04.01...04.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.05.01...05.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.06.01...06.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.07.01...07.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.08.01...08.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.09.01...09.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.10.01...10.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.11.01...11.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_lst_mod11a2.nighttime.trend_p50_1km_s0_std'] = \
        merged[['clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.01.01...01.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.02.01...02.28',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.03.01...03.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.04.01...04.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.05.01...05.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.06.01...06.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.07.01...07.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.08.01...08.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.09.01...09.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.10.01...10.31',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.11.01...11.30',
             'clm_lst_mod11a2.nighttime.trend_p50_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.01.01...01.31_mean'] = \
        merged[['clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.01.01...01.31_std'] = \
        merged[['clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p05_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.01.01...01.31_mean'] = \
        merged[['clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.01.01...01.31_std'] = \
        merged[['clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p50_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.01.01...01.31_mean'] = \
        merged[['clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.12.01...12.31']].mean(axis=1)

    merged['clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.01.01...01.31_std'] = \
        merged[['clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.01.01...01.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.02.01...02.28',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.03.01...03.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.04.01...04.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.05.01...05.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.06.01...06.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.07.01...07.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.08.01...08.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.09.01...09.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.10.01...10.31',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.11.01...11.30',
                'clm_lst_mod11a2.nighttime_p95_1km_s0..0cm_.12.01...12.31']].std(axis=1)

    merged['lcv_blue_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_blue_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_blue_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_blue_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_blue_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_blue_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_blue_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_blue_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_blue_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_blue_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_blue_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_blue_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_green_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_green_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_green_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_green_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_green_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_green_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_green_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_green_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_green_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_green_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_green_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_green_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_red_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_red_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_red_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_red_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_ndvi_landsat.glad.tmwm_p75.03.21...06.24'] = (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24']) /  (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] + merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'])
    merged['lcv_ndvi_landsat.glad.tmwm_p25.03.21...06.24'] = (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']) /  (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24'] + merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24'])
    merged['lcv_ndvi_landsat.glad.tmwm_p50.03.21...06.24'] = (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24']) /  (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24'] + merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24'])
    merged['lcv_ndvi_landsat.glad.tmwm_p75_p25.03.21...06.24'] = merged['lcv_ndvi_landsat.glad.tmwm_p75.03.21...06.24'] - merged['lcv_ndvi_landsat.glad.tmwm_p25.03.21...06.24']

    merged['lcv_ndvi_landsat.glad.tmwm_p75.06.25...09.12'] = (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12']) /  (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] + merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'])
    merged['lcv_ndvi_landsat.glad.tmwm_p25.06.25...09.12'] = (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']) /  (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12'] + merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12'])
    merged['lcv_ndvi_landsat.glad.tmwm_p50.06.25...09.12'] = (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12']) /  (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] + merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'])
    merged['lcv_ndvi_landsat.glad.tmwm_p75_p25.06.25...09.12'] = merged['lcv_ndvi_landsat.glad.tmwm_p75.06.25...09.12'] - merged['lcv_ndvi_landsat.glad.tmwm_p25.06.25...09.12']

    merged['lcv_ndvi_landsat.glad.tmwm_p75.09.13...12.01'] = (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01']) /  (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] + merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'])
    merged['lcv_ndvi_landsat.glad.tmwm_p25.09.13...12.01'] = (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']) /  (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01'] + merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01'])
    merged['lcv_ndvi_landsat.glad.tmwm_p50.09.13...12.01'] = (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01']) /  (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] + merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'])
    merged['lcv_ndvi_landsat.glad.tmwm_p75_p25.09.13...12.01'] = merged['lcv_ndvi_landsat.glad.tmwm_p75.09.13...12.01'] - merged['lcv_ndvi_landsat.glad.tmwm_p25.09.13...12.01']

    merged['lcv_ndvi_landsat.glad.tmwm_p75.12.02...03.20'] = (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20']) /  (merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] + merged['lcv_red_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'])
    merged['lcv_ndvi_landsat.glad.tmwm_p25.12.02...03.20'] = (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20'] - merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']) /  (merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20'] + merged['lcv_red_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20'])
    merged['lcv_ndvi_landsat.glad.tmwm_p50.12.02...03.20'] = (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20']) /  (merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] + merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'])
    merged['lcv_ndvi_landsat.glad.tmwm_p75_p25.12.02...03.20'] = merged['lcv_ndvi_landsat.glad.tmwm_p75.12.02...03.20'] - merged['lcv_ndvi_landsat.glad.tmwm_p25.12.02...03.20']

    merged['lcv_ndvi_seasonal_change_1'] = merged['lcv_ndvi_landsat.glad.tmwm_p50.12.02...03.20'] - merged['lcv_ndvi_landsat.glad.tmwm_p50.09.13...12.01']
    merged['lcv_ndvi_seasonal_change_2'] = merged['lcv_ndvi_landsat.glad.tmwm_p50.09.13...12.01'] - merged['lcv_ndvi_landsat.glad.tmwm_p50.06.25...09.12']
    merged['lcv_ndvi_seasonal_change_3'] = merged['lcv_ndvi_landsat.glad.tmwm_p50.06.25...09.12'] - merged['lcv_ndvi_landsat.glad.tmwm_p50.03.21...06.24']
    merged['lcv_ndvi_seasonal_change_4'] = merged['lcv_ndvi_landsat.glad.tmwm_p50.03.21...06.24'] - merged['lcv_ndvi_landsat.glad.tmwm_p50.12.02...03.20']

    merged['lcv_nir_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_nir_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_nir_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_nir_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_nir_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_nir_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_spring'] = merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] - merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24']
    merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_summer'] = merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] - merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12']
    merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_winter'] = merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] - merged['lcv_nir_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01']

    merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_spring'] = merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] - merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24']
    merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_summer'] = merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] - merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12']
    merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_winter'] = merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] - merged['lcv_blue_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01']

    merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_spring'] = merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24']
    merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_summer'] = merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12']
    merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_winter'] = merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] - merged['lcv_red_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01']

    merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_spring'] = merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12'] - merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.03.21...06.24']
    merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_summer'] = merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01'] - merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.06.25...09.12']
    merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_seasonal_winter'] = merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.12.02...03.20'] - merged['lcv_green_landsat.glad.tmwm_p50_30m_0..0cm_.09.13...12.01']

    merged['lcv_swir1_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_swir1_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_swir1_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_swir1_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_swir1_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_swir1_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_swir1_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_swir1_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_swir1_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_swir1_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_swir1_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_swir1_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']
    merged['lcv_swir1_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_swir1_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_swir1_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_swir2_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_swir2_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_swir2_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_swir2_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_swir2_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_swir2_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_swir2_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_swir2_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_swir2_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_swir2_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_swir2_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_swir2_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']
    merged['lcv_swir2_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_swir2_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_swir2_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    merged['lcv_thermal_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.03.21...06.24'] = merged['lcv_thermal_landsat.glad.tmwm_p75_30m_0..0cm_.03.21...06.24'] - merged['lcv_thermal_landsat.glad.tmwm_p25_30m_0..0cm_.03.21...06.24']
    merged['lcv_thermal_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.06.25...09.12'] = merged['lcv_thermal_landsat.glad.tmwm_p75_30m_0..0cm_.06.25...09.12'] - merged['lcv_thermal_landsat.glad.tmwm_p25_30m_0..0cm_.06.25...09.12']
    merged['lcv_thermal_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.09.13...12.01'] = merged['lcv_thermal_landsat.glad.tmwm_p75_30m_0..0cm_.09.13...12.01'] - merged['lcv_thermal_landsat.glad.tmwm_p25_30m_0..0cm_.09.13...12.01']
    merged['lcv_thermal_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_thermal_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_thermal_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']
    merged['lcv_thermal_landsat.glad.tmwm_p25_p_75_30m_0..0cm_.12.02...03.20'] = merged['lcv_thermal_landsat.glad.tmwm_p75_30m_0..0cm_.12.02...03.20'] - merged['lcv_thermal_landsat.glad.tmwm_p25_30m_0..0cm_.12.02...03.20']

    #scale to same range
    sc = StandardScaler()
    #exclude class form scaling
    column_class_copy = copy.deepcopy(merged['class'])
    rescaled = merged.drop('class', axis=1)
    column_names = rescaled.columns.values
    rescaled = sc.fit_transform(rescaled)
    rescaled = pd.DataFrame(data=rescaled,columns=column_names)
    rescaled['class'] = column_class_copy.values
    return rescaled

def split_set(merged, rs = 42):
    # Split the dataset
    Tr_X, Te_X, Tr_Y, Te_Y = train_test_split(merged.drop(columns=["class"]), merged["class"],
            train_size=0.7, random_state = rs, stratify = merged["class"])
    return Tr_X, Te_X, Tr_Y, Te_Y

def feature_selection(merged,method = 'rfe'):
    class_copy = merged['class']
    Tr_X, Te_X, Tr_Y, Te_Y = split_set(merged, rs=31)

    if method == 'rfe':
        # #Selecting the Best important features according to Logistic Regression using SelectFromModel
        selector = SelectFromModel(estimator=LogisticRegression())

    elif method == 'sfs':
        # Selecting the Best important features according to Logistic Regression
        selector = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select=200,cv=10 ,direction='backward')

    else:
        selector = SelectKBest(f_classif, k=50)

    selector.fit(Tr_X, Tr_Y)
    selected_feature_names = Tr_X.columns[selector.get_support()]
    # print(selected_features)
    selected_features = merged[selected_feature_names]

    if not 'class' in selected_features.columns:
        selected_features['class'] = class_copy

    return selected_features