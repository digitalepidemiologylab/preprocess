import os
import sys
import json
from contextlib import contextmanager
import numpy as np
import hashlib
import pandas as pd

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)

def get_df_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
