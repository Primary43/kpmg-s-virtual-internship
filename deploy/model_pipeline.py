import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class DataPreprocessor:
    def __init__(self, job_cluster_path='data/job_cluster_df.csv'):
        self.job_cluster_path = job_cluster_path
        self.ordinal_encoder = OrdinalEncoder(categories=[['Mass Customer', 'High Net Worth', 'Affluent Customer']])
        self.job_cluster_df = pd.read_csv(job_cluster_path, index_col=0)

    def add_log_age_tenure(self, df):
        df['Log_age_tenure'] = np.log(df['age'] + 1) + np.log(df['tenure'] + 1)
        df = df.drop(['age', 'tenure'], axis=1)
        return df

    def encode_job_cluster(self, df):
        df = df.merge(self.job_cluster_df, on='job_title', how='left')
        df['job_cluster'] = df['job_cluster'].fillna(41.0)
        df = df.drop(['job_title'], axis=1)
        return df

    def ordinal_encode(self, df):
        df['wealth_segment'] = self.ordinal_encoder.fit_transform(df[['wealth_segment']])
        return df

    def create_dummy_variables(self, df):
        categorical_cols = ['gender', 'state', 'job_industry_category', 'owns_car', 'job_cluster']
        df = pd.get_dummies(df, columns=categorical_cols)

        complete_dummy_columns = ['wealth_segment', 'past_3_years_bike_related_purchases',
                                   'property_valuation', 'Log_age_tenure', 'gender_Female', 'gender_Male',
                                   'state_NSW', 'state_QLD', 'state_VIC',
                                   'job_industry_category_Argiculture',
                                   'job_industry_category_Entertainment',
                                   'job_industry_category_Financial Services',
                                   'job_industry_category_Health', 'job_industry_category_IT',
                                   'job_industry_category_Manufacturing', 'job_industry_category_Property',
                                   'job_industry_category_Retail',
                                   'job_industry_category_Telecommunications', 'owns_car_False',
                                   'owns_car_True', 'job_cluster_0.0', 'job_cluster_1.0',
                                   'job_cluster_2.0', 'job_cluster_3.0', 'job_cluster_4.0',
                                   'job_cluster_5.0', 'job_cluster_6.0', 'job_cluster_7.0',
                                   'job_cluster_8.0', 'job_cluster_9.0', 'job_cluster_10.0',
                                   'job_cluster_11.0', 'job_cluster_12.0', 'job_cluster_13.0',
                                   'job_cluster_14.0', 'job_cluster_15.0', 'job_cluster_16.0',
                                   'job_cluster_17.0', 'job_cluster_18.0', 'job_cluster_19.0',
                                   'job_cluster_20.0', 'job_cluster_21.0', 'job_cluster_22.0',
                                   'job_cluster_23.0', 'job_cluster_24.0', 'job_cluster_25.0',
                                   'job_cluster_26.0', 'job_cluster_27.0', 'job_cluster_28.0',
                                   'job_cluster_29.0', 'job_cluster_30.0', 'job_cluster_31.0',
                                   'job_cluster_32.0', 'job_cluster_33.0', 'job_cluster_34.0',
                                   'job_cluster_35.0', 'job_cluster_36.0', 'job_cluster_37.0',
                                   'job_cluster_38.0', 'job_cluster_39.0', 'job_cluster_40.0']
        # Reindex the DataFrame to include all expected dummy columns, filling missing ones with 0
        df = df.reindex(columns=complete_dummy_columns, fill_value=0)

        # Cast to uint8 for consistency
        df = df.astype('uint8')

        # Drop unwanted columns, if they exist
        columns_to_drop = ['job_industry_category_Unknown', 'job_cluster_41.0']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
        df = df[complete_dummy_columns]
        return df

    def reformat(self, df):
        df.index.name = 'customer_id'
        df[['past_3_years_bike_related_purchases', 'property_valuation']] = df[['past_3_years_bike_related_purchases', 'property_valuation']].astype('float64')
        float_col = df.select_dtypes(include=['float64']).columns
        for col in float_col:
            df[col] = df[col].apply(lambda x: f"{x:.6f}")
            df[col] = df[col].astype(float)
        return df

    def process(self, df):
        original_index = df.index
        df = self.add_log_age_tenure(df)
        df = self.encode_job_cluster(df)
        df = self.ordinal_encode(df)
        df = self.create_dummy_variables(df)
        df = self.reformat(df)
        df.index = original_index
        return df

