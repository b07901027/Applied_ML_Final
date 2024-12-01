import pandas as pd
import time
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

N_components = 40 # Number of components to keep
N_batches = 10 # Increase if running too slow
dropped_columns = ['id_33', 'id_34']

# read cli arguments, clean input
improper_input_reason = ''
if len(sys.argv) == 5:
    train_X_filename = sys.argv[1]
    test_X_filename = sys.argv[2]
    train_X_output_filename = sys.argv[3]
    test_X_output_filename = sys.argv[4]
elif len(sys.argv) == 3:
    train_X_filename = sys.argv[1]
    test_X_filename = None
    train_X_output_filename = sys.argv[2]
    test_X_output_filename = None
else:
    print('To feature engineer, include 2/4 filenames as sys args (or import and call the function):\n\tfilename 1: combined and filled train data\n\tfilename 2: combined and filled test data (optional)\n\tfilename 3: train output filename\n\tfilename 4: test output filename (optional)')
    exit()

t1 = time.time()

# Read in provided csvs
print('Reading in...')
sys.stdout.flush()
t1 = time.time()
df_train = pd.read_csv(train_X_filename)
df_test = pd.read_csv(test_X_filename) if test_X_filename is not None else None

# Standardize column names
print('Standardizing column names...')
df_train.columns = df_train.columns.str.replace('-', '_')
if df_test is not None:
    df_test.columns = df_test.columns.str.replace('-', '_')

# Drop columns with NaN values
print('Dropping columns...')
df_train = df_train.drop(columns=dropped_columns, errors='ignore')
if df_test is not None:
    df_test = df_test.drop(columns=dropped_columns, errors='ignore') 

# Normalize
print('Normalizing...')
scaler = StandardScaler()
scaler.fit(df_train)
df_train = scaler.transform(df_train)
if df_test is not None:
    df_test = scaler.transform(df_test)

# # PCA
# print('PCA...')
# sys.stdout.flush()
# pca = PCA(n_components=N_components)
# pca.fit(df_train)
# df_train = pca.transform(df_train)
# df_test = pca.transform(df_test)

# IPCA (Incremental PCA) for efficiency
print('IPCA...')
sys.stdout.flush()
pca = IncrementalPCA(n_components=N_components)
batch_size = int(len(df_train)/N_batches)
for i in range(N_batches):
    print(f'Batch {i+1}/{N_batches}')
    sys.stdout.flush()
    pca.partial_fit(df_train[i*batch_size:(i+1)*batch_size])
print('Transforming...')
df_train = pca.transform(df_train)
if df_test is not None:
    df_test = pca.transform(df_test)

# Write output
print('Writing output...')
sys.stdout.flush()
pd.DataFrame(df_train).to_csv(train_X_output_filename)
if test_X_output_filename:
    pd.DataFrame(df_test).to_csv(test_X_output_filename)

# Plot explained variance
print('Plotting explained variance...')
sys.stdout.flush()
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.savefig('explained_variance_ratio.png')

# Plot cumulative explained variance
print('Plotting cumulative explained variance...')
sys.stdout.flush()
plt.figure()
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title('PCA Cumulative Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.ylim(0,1)
plt.savefig('cumulative_explained_variance_ratio.png')


# Print first few component variance ratios
print('First few component variance ratios:')
print(pca.explained_variance_ratio_[:max(N_components,20)])

t2 = time.time()
print(f'Finished Feature Engineering, elapsed time: {t2-t1}s')
sys.stdout.flush()
