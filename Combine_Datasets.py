# Combines datasets

import pandas as pd
from Format_csvs import *
# from Combine_Analyze_csvs import *
import os
import time

# read cli arguments, clean input
improper_input_reason = ''
if len(sys.argv) == 4:
    transaction_filename= sys.argv[1]
    identity_filename = sys.argv[2]
    output_filename = sys.argv[3]
else:
    print('To output dataframes, include 3 filenames as sys args (or import and call the function):'+
          '\n\tfilename 1: transaction csv filename'+
          '\n\tfilename 2: identity csv filename'
          '\n\tfilename 3: output filename'
          )
    exit()



read_Cols(transaction_filename=transaction_filename,
          id_filename=identity_filename,
          combined_filename=output_filename,
          non_numerical_filename='',
          verbose=False)