import os
import pandas as pd

PROJ_DIR = os.path.dirname(__file__)

lda_output_csv = os.path.join(PROJ_DIR, 'data', 'PROTO_lda.csv')
Twitter_output_csv = os.path.join(PROJ_DIR, 'data', 'PROTO_merged.csv')
output_csv = 'PROTO_merged_lda.csv'
topics_df = pd.read_csv(lda_output_csv)
Twitter_df = pd.read_csv(Twitter_output_csv)
topics_df.rename(columns={'user': 'screen_name'},
                 inplace=True)
ensemble_ready_df = pd.merge(Twitter_df, topics_df, on='screen_name')
ensemble_ready_df = ensemble_ready_df.rename(columns={'followers': 'u_followers_count',
                                                      'status_count': 'u_statuses_count'})
ensemble_ready_df.to_csv(output_csv)
