## Workflow

Combine_Datasets.py -> combine identity and transaction datasets
Fill_Dataset_NA.py -> fills NA values

to change strategy to fill numerical values, edit the format_numerical() class method in Format_csvs.py

I have already generated 2 datasets(uploaded to google drive folder). For each dataset There is a given X, Y, and Changelog(tracks what categorical variables are transformed into)
    - Large dataset: All non-numerics one hard encoded, NO POOLING OF EACH COLUMN (Many hundreds more features)
    - Reduced dataset: All non-numerics one hard encoded, top couple categories used. Others assigned to an 'other' label

    

Unique_values.csv gives the unique values found in each non numerical variable
