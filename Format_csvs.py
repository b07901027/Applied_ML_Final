# Helper functions to fill/format CSV values

import pandas as pd
import numpy as np
import csv
import sys
import math
import time

'''
sparse: >98% missing
Reading the cols:
    id_12 - Found/notfound, set all NA to not found
    id_15 - new/found/unknown, set all NA to unknown
    id_16 - Found/notfound, set all NA to not found
    id_23 - IP_PROXY:TRANSPARENT, IP_PROXY:ANONYMOUS, IP_PROXY:HIDDEN:, sparse
    id_27 - Found/notfound, set all NA to not found, sparse
    id_28 - Found/new, new category for NA
    id_29 - Found/notfound, set all NA to not found
    id_30 - OS, categorize by company? how updated?
    id_31 - browser type
    id_33 - screen resolution
    id_34 - match_status: #, hash is 0, 1, 2, some other numbers
    id_35 - T/F, new catagory for NA
    id_36 - T/F, new catagory for NA
    id_37 - T/F, new catagory for NA
    id_38 - T/F, new catagory for NA
    DeviceType - mobile vs desktop, can likely use this info plus id_30-31 to reduce dims
    DeviceInfo - device model number
    ProductCD - one letter
    card4 - card company
    card6 - credit/debit/credit or debit/other - can probably be encoded, set NA to credit or debit
    P_emaildomain - email domains
    R_emaildomain - email domains
    M1 - T/F, new catagory for NA
    M2 - T/F, new catagory for NA
    M3 - T/F, new catagory for NA
    M4 - T/F, new catagory for NA
    M5 - T/F, new catagory for NA
    M6 - T/F, new catagory for NA
    M7 - T/F, new catagory for NA
    M8 - T/F, new catagory for NA
    M9 - T/F, new catagory for NA 
'''


# A dict of identifiable sets that are a subset of the possible non numerical values
default_processing_categories = {
    'T_F_labels': {'T', 'F'},
    'F_NF_labels': {'Found', 'NotFound'},
    'N_F_U_labels': {'New', 'Found', 'Unknown'},
    'F_N_labels': {'Found, New'},
    'IP_labels': {'IP_PROXY:TRANSPARENT', 'IP_PROXY:ANONYMOUS', 'IP_PROXY:HIDDEN'},
    'match_status': {'match_status:1'},
    'browser_types': {'edge'},
    'OS_types': {'Windows 10'},
    'screen_resolution': {'1920x1080'},
    'device_type': {'mobile'},
    'device_info': {'Windows', 'MacOs'},
    'product_CD': {'W', 'C', 'R'},
    'card4': {'visa'},
    'card6': {'credit', 'debit'},
    'email_domain': {'gmail.com'},
    'Device_Info_types': {'iOS Device', 'Windows', 'MacOS'},
    }

# Column names that are encoded with no preprocessing
default_one_hot_list = ['T_F_labels', 'F_N_labels', 'IP_labels', 'device_type']



def read_Cols(transaction_filename='train_transaction.csv', id_filename='train_identity.csv', combined_filename='', non_numerical_filename='', verbose=False):
    
    # Combines given filenames for identity and transaction csvs
    # INPUTS:
        # transaction_filename: filename of the transaction csv (input)
        # id_filename: filename of the identity csv (input)
        # combined_filename: filename of the combined dataset (output)
        # non_numerical_filename: filename of the non-numerical cols of the dataset (output, optional)
    # Returns a dict with keys:
    #   combined_df: the combined dataframe
    #   non_numerical_cols: a list of column names that have non numerical data


    # Read in provided csvs
    print('Reading in...')
    sys.stdout.flush()
    t1 = time.time()
    df_trans = pd.read_csv(transaction_filename)
    df_id = pd.read_csv(id_filename)


    # Combine csvs
    print('Packaging Data...')
    sys.stdout.flush()
    t2 = time.time()
    df_total = pd.merge(df_id, df_trans, how='outer')
    X_train = df_total.drop(columns=['TransactionID', 'isFraud'] if 'isFraud' in df_total.columns else ['TransactionID'])
    Y_train = df_total['isFraud'] if 'isFraud' in df_total.columns else None


    # Pull non numerical columns
    print('Analyzing columns...')
    sys.stdout.flush()
    t3 = time.time()
    print('cols in X, types ')
    non_numerical_cols = get_non_numerical_cols(df_total, verbose=False)



    # Export combined sets to new files
    t4 = time.time()
    if combined_filename:
        df_total.to_csv(combined_filename)
        print(f'Combined data exported to {combined_filename}')
    if non_numerical_filename:
        output_df = df_total[non_numerical_cols]
        output_df.to_csv(non_numerical_filename)
        print(f'Non-numerical columns exported to {non_numerical_filename}')

    # Print output
    print(df_total.head())
    print(f'Elapsed Reading time: {t2-t1:.2f}s')
    print(f'Elapsed Data manipulation time: {t3-t2:.2f}s')
    print(f'Elapsed Column typing time: {t4-t3:.2f}s')

    return {'combined_df': df_total, 'non_numerical_cols': non_numerical_cols}        


# given a dataframe, returns columns that have non numerical data 
def get_non_numerical_cols(df, verbose=False):
    non_numerical_cols = []
    for col in df.columns:
        data_type = set(type(i) for i in df[col])
        if data_type.__contains__(type(' ')):
            non_numerical_cols.append(col)
        
        if verbose:
            print(f"\t{col}: {data_type}")
    return non_numerical_cols

# Class to clean data, fill NA values
class ColumnFormatter:
    def __init__(self, processing_categories=default_processing_categories, apply_one_hot_list=default_one_hot_list, verbose=False):
        # INPUT: 
        #   processing_categories: a dictionary with a value of a distinct subset of values that require a preprocessing strategy
        #   apply_one_hot_list: A list of column names that are one-hot encoded with no preprocessing
        #   verbose: dump progress/debug data to console[T/F]

        
        self.processing_categories = processing_categories
        self.apply_one_hot_list = apply_one_hot_list

        # Stores what variables get transformed into for one hot encoding
        self.variable_output_translation = {}
        self.verbose = verbose # print debug data or not

    
    # Format all non numerical cols
    def format_non_numerical(self, df_input=None, non_numerical_columns=[], filename=''):
        self.vprint(f'\npassed non_numerical_columns: {non_numerical_columns}')
        
        # Get dataframe from either passed df or csv
        if filename:
            df = pd.read_csv(filename)
        else:
            df = df_input
        self.vprint('\tFinished instantiating df')
        
        # Format columns
        for col in non_numerical_columns:
            values = set(df[col])
            self.vprint(f'\tformatting col: {col}')

            # Check if value is in any of the sets that get one hot encoding applied by default
            direct_one_hot_condition = False
            for set_name in self.apply_one_hot_list:
                category_set = self.processing_categories[set_name]
                direct_one_hot_condition = direct_one_hot_condition or values.issuperset(category_set)

            self.vprint(f'\t\tChecking col category: ', end='')

            # Check if column values belong to any preprocessing category else one hot encode
            if direct_one_hot_condition:
                self.vprint(f'apply_one_hot')
                df = self.one_hot_encode(df, col)

            elif values.issuperset(self.processing_categories['F_NF_labels']):
                self.vprint(f'F_NF_labels')
                df[col] = df[col].fillna('NotFound')
                df = self.one_hot_encode(df, col)

            elif values.issuperset(self.processing_categories['N_F_U_labels']):
                self.vprint(f'N_F_U_labels')
                df[col] = df[col].fillna('Unknown')
                df = self.one_hot_encode(df, col)

            elif values.issuperset(self.processing_categories['match_status']):
                self.vprint(f'match_status')
                df[col] = df[col].transform(trim_match_status)
                self.variable_output_translation['match_status'] = "stripped \'match_status:\'"

            elif values.issuperset(self.processing_categories['browser_types']):
                self.vprint(f'browser_types')
                df[col] = df[col].transform(categorize_browser_types)
                df = self.one_hot_encode(df, col)

            elif values.issuperset(self.processing_categories['OS_types']):
                self.vprint(f'OS_types')                
                df[col] = df[col].transform(categorize_OS_types)
                df = self.one_hot_encode(df, col)

            elif values.issuperset(self.processing_categories['screen_resolution']):
                self.vprint(f'screen_resolution')
                df[col] = df[col].transform(get_num_pixels)
                self.variable_output_translation['match_status'] = "stripped \'match_status:\'"
            elif values.issuperset(self.processing_categories['Device_Info_types']):
                self.vprint(f'Device_Info_types')                
                df[col] = df[col].transform(categorize_Device_Info_types)
                df = self.one_hot_encode(df, col)
            elif values.issuperset(self.processing_categories['email_domain']):
                self.vprint(f'Device_Info_types')                
                df[col] = df[col].transform(categorize_email_domain_types)
                df = self.one_hot_encode(df, col)
            else:
                self.vprint(f'default case')
                df = self.one_hot_encode(df, col)
            
        # Output
        if filename:
            df.to_csv(filename)
            print(f'Processed non numerical data exported to {filename}')
        return df

        
    # passes back a df that replaces the given column name with new one-hot encoded columns
    def one_hot_encode(self,df_input, col, verbose=False):
        self.vprint(f'\t\tOHE on col: {col}')
        df = df_input.copy()

        # Gets a non repeating list of all variable values in the column and puts NA at the front if it is included. Otherwise sorts the elements
        values = list(set(df[col]))
        # values.insert(0, '')
        num_bits = len(values) - 1
        self.vprint(f'\t\tValues: {values}')

        if any(isinstance(i, float) and math.isnan(i) for i in values):
            self.vprint('has nans---------------------------------------------------------------------------------------------')
            nan_index = next(i for i, v in enumerate(values) if isinstance(v, float) and math.isnan(v))
            NA = values.pop(nan_index)
            values.insert(0, NA)
            
        else:
            try:
                values.sort()
            except:
                print(f'error sorting col: {col}')
                sys.stdout.flush()
        self.vprint(f'\t\tList of vals instantiated')

        # Create the 1 hot encoding name columns
        new_col_names = [f'{col}_bit_{i}' for i in range(num_bits)]
        
        # Create dict to translate between non-numerical data and bits to turn on with the first value in the list used as the default case
        # Also adds to a dict that tracks what non-numerical variables were assigned to what bit sequence (combination of bits in cols)
        assign_bits_dict = {}
        for i, value in enumerate(values):
            bit_sequence = np.array([0]*(num_bits))
            if i != 0:
                bit_sequence[i-1] = 1
            assign_bits_dict[value] = bit_sequence
            self.variable_output_translation[f'{col}: {value}'] = str(bit_sequence)

        self.vprint(f'\t\tTranslation dicts created')

        ## Fill in the new one hot encoded columns
        # Match each value to its corresponding bit string
        new_df_data = np.vstack(df[col].map(assign_bits_dict).to_numpy())

        # Transform row-wise bit strings into column-wise bit strings
        new_df_data = np.transpose(new_df_data)
        self.vprint(f'\t\tNew one hot cols encoded')

        # Assign these bit strings to the new named columns
        for i, bit_col in enumerate(new_df_data):
            pd.concat([df, pd.DataFrame(data=bit_col, columns=[new_col_names[i]])], axis=1)

        # Discard original column
        df = df.drop(col, axis=1)

        # DEBUG
        if verbose:
            print(f'DEBUG DF: {df}')
            print(f'new col names: {new_col_names}')
            print(f'assign_bits_dict: {assign_bits_dict}')
            print(f'values: {values}')
            print(f'assign_bits_dict: {assign_bits_dict}')
            print(f'new_df_data: {new_df_data}')
        
        return df

    # Formats Numerical columns
    def format_numerical(self, df_input, numerical_cols, filename=''):
        df = df_input.copy()
        for col in numerical_cols:

            # Just assigning all to mean for now, edit this section if you want to do another method
            col_mean_val = df[col].dropna().mean()
            df[col] = df[col].fillna(col_mean_val)
            
        # Output
        if filename:
            df.to_csv(filename)
            print(f'Processed numerical data exported to {filename}')   

    # Write the log of translations made
    def write_output(self, filename):
        # Write file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Original', 'Bit Sequence'])
            
            # Write each key-value pair as a row
            for key, value in self.variable_output_translation.items():
                writer.writerow([key, value])

    # Only print if verbose set to true
    def vprint(self, string, end='\n'):
        if self.verbose:
            print(string, end=end)
            sys.stdout.flush()
        else:
            pass

# Helper function to transform browser names
def categorize_browser_types(browser):
                browser_dict = {'mobile safari': 'mobile safari', 'safari':'safari', 'chrome':'chrome', 'edge': 'edge', 'firefox': 'firefox'}
                try:
                    for b in browser_dict:
                        if b in browser.lower():
                            return browser_dict[browser]
                except:
                    pass    
                return 'other'

# Helper function to transform OS names
def categorize_OS_types(OS):
                
                OS_dict = {'os': 'ios', 'windows': 'windows'}
                try:
                    for o in OS_dict:
                        if o in OS.lower():
                            return OS_dict[OS]
                except:
                    pass   
                return 'other'

# Helper function to transform resolutions to pixel count
def get_num_pixels(str_resolution):
                    if isinstance(str_resolution, float) and math.isnan(str_resolution):
                        return str_resolution
                    
                    resolution = str_resolution.split('x') 
                    return float(resolution[0])*float(resolution[1])

# Helper function to format the match status column
def trim_match_status(input):

    try:
        return float(input.replace('match_status:'))
    except:
         return input 
    
# Helper function to transform Device Info
def categorize_Device_Info_types(Device_info):
                
                Device_Info_types = {'macos': 'macos', 'ios': 'ios', 'windows': 'windows'}
                try:
                    for o in Device_Info_types:
                        if o in Device_info.lower():
                            return Device_Info_types[Device_info]
                except:
                    pass   
                return 'other'

# Helper function to transform Email Domains
def categorize_email_domain_types(Email):
                
                Email_domain_types = {'gmail': 'gmail', 'yahoo': 'yahoo', 'hotmail': 'hotmail', 'outlook': 'outlook'}
                try:
                    for e in Email_domain_types:
                        if e in Email.lower():
                            return Email_domain_types[Email]
                except:
                    pass   
                return 'other'




