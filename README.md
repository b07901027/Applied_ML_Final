## Workflow

Combine_Datasets.py -> combine identity and transaction datasets
Fill_Dataset_NA.py -> fills NA values

to change strategy to fill numerical values, edit the format_numerical() class method in Format_csvs.py

I have already generated 2 datasets(uploaded to google drive folder). For each dataset There is a given X, Y, and Changelog(tracks what categorical variables are transformed into)
    - Large dataset: All non-numerics one hard encoded, NO POOLING OF EACH COLUMN (Many hundreds more features)
    - Reduced dataset: All non-numerics one hard encoded, top couple categories used. Others assigned to an 'other' label

    

Unique_values.csv gives the unique values found in each non numerical variable

Model:

input file:

X_Reduced_set.csv / X_Reduced_PCA_set.csv (only used in DNN.py SMOTE.py for gridCV search, since gridCV takes too long to complete)

Y_Reduced_set.csv

from Jacob's preprocessing data

Designed model:

use X_Reduced_PCA_set.csv

1. DNN: use gridCV to search best parameter -- metrics: 0.88
2. SMOTE: deal with unbalanced dataset --metrics: 0.86-0.88

use X_Reduced_set.csv

4. XGB: --metrics: 0.91
5. LGBM: --metrics: 0.91
6. CAT: --metrics: 0.92
7. ensemble_voting: combine XGB, LGBM, CATusing voting--metrics: 0.94
8. ensemble_voting_weighted: combine XGB, LGBM, CAT using voting with emphasis on minority group -- metrics:0.95
9. ensemble_kcross_stacking: combine XGB, LGBM, CAT using stacking with emphasis on minority group, and improve accuracy with kcross -- metrics:0.96

