# tcrdistgpu

## Credit

* The idea for a GPU accerlated TCRdist and initial code -- mikhail.pogorelyy@stjude.org (Mikhail Pogorelyy).
* Python package and implemenation CPU, GPU accerlated TCRdist -- kmayerbl@fredhutch.org (Koshlan Mayer-Blackwell)


### Limitations

Currently only works for human TCR data.


### Install

```
pip install git+https://kmayerb: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@github.com/kmayerb/tcrdistgpu.git
```



## Basics 

1. If using T4 instance on google collab, use `cuda`, otherwise use `cpu` mode
2. Specify chunk size (i.e, number of rows per intermediate result)
3. Provide column names for required columns matching headers in your data. 
The defaults are [`'cdr3a','cdr3b','vb','va']`, but you do not have to reformat
you data if you specify `cdr3b_col` ,`cdr3a_col` , `vb_col`, `va_col` arguments

See example below for functionality:


```python
import sys
sys.path.append('/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/')
import os
import pandas as pd 
from tcrdistgpu.distance import TCRgpu 
import numpy as np
package_data = '/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/tcrdistgpu/data'
data = pd.read_csv(os.path.join(package_data, 'dash_human.csv'))
tg = TCRgpu(mode = 'cpu', 
                    chunk_size = 50,
                    cdr3b_col = 'cdr3_b_aa',
                    cdr3a_col = 'cdr3_a_aa',
                    vb_col = 'v_b_gene',
                    va_col = 'v_a_gene')

# Vectorize data
encoding          = tg.encode_tcrs(data) 

# Get K nearest neighbors (KNN) and KNN dists
knn_idx, knn_dist = tg.compute(encoding, encoding, max_k = 20)

# Get probability mass functions in each bin e.g., [0, 12),[12,24) not including upper edge
bins = np.arange(0, 401, 12)
knn_pmf  = tg.compute_distribution(encoding, encoding, max_k = 20, pmf = True, bins = bins)  
all_pmf  = tg.compute_distribution(encoding, encoding, pmf = True, bins = bins)
# Get counts in each bin
knn_hist  = tg.compute_distribution(encoding, encoding, max_k = 20, pmf = False, bins = bins)  
all_hist  = tg.compute_distribution(encoding, encoding, pmf = False, bins = bins)

# Compute array (small only, no compression)
arr_dists           = tg.compute_array(encoding, encoding)

# Get CSR spare matrics
# knn retained
csr_dists_max_k     = tg.compute_csr(encoding, encoding, max_k = 10)
# radius search retained [0, max_dist] (i.e., includes max_dist)
csr_dists_max_dist  = tg.compute_csr(encoding, encoding, max_dist = 100)
```

## Easily compare 2 encodings 

```python 
encoding1       = tg.encode_tcrs(data.query('epitope == "M1"').reset_index(drop = True))

# Get K nearest neighbors (KNN) and KNN dists
knn_idx, knn_dist = tg.compute(encoding1, encoding, max_k = 20)

# Get probability mass functions in each bin e.g., [0, 12),[12,24) not including upper edge
bins = np.arange(0, 401, 12)
knn_pmf  = tg.compute_distribution(encoding1, encoding, max_k = 20, pmf = True, bins = bins)  
all_pmf  = tg.compute_distribution(encoding1, encoding, pmf = True, bins = bins)
# Get counts in each bin
knn_hist  = tg.compute_distribution(encoding1, encoding, max_k = 20, pmf = False, bins = bins)  
all_hist  = tg.compute_distribution(encoding1, encoding, pmf = False, bins = bins)

# Compute array (small only, no compression)
arr_dists           = tg.compute_array(encoding1, encoding)

# Get CSR spare matrics
# knn retained
csr_dists_max_k     = tg.compute_csr(encoding1, encoding, max_k = 10)
# radius search retained [0, max_dist] (i.e., includes max_dist)
csr_dists_max_dist  = tg.compute_csr(encoding1, encoding, max_dist = 100)

```

## KNN-classification in one function

```python

import sys
sys.path.append('/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/')
package_data = '/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/tcrdistgpu/data'
import os
import pandas as pd 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tcrdistgpu.knn import knn_tcr



data = pd.read_csv(os.path.join(package_data, 'dash_human.csv'))[['cdr3_b_aa', 'v_b_gene','cdr3_a_aa', 'v_a_gene','epitope']]

y = (data['epitope'] == "M1").astype(int).values
X = data[['cdr3_b_aa', 'v_b_gene','cdr3_a_aa', 'v_a_gene']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df, prob = knn_tcr(tcr_train = X_train,
        tcr_test = X_test,
        label_train = y_train,
        label_test = y_test,
        chain = "b", 
        mode = "cpu", 
        kbest = 20,
        krange = range(1,19,1),
        adjust_class_weights = True,
        cdr3b_col = 'cdr3_b_aa',
        cdr3a_col = 'cdr3_a_aa',
        vb_col = 'v_b_gene',
        va_col = 'v_a_gene')

print(df)

df, prob = knn_tcr(tcr_train = X_train,
        tcr_test = X_test,
        label_train = y_train,
        label_test = y_test,
        chain = "ab", 
        mode = "cpu", 
        kbest = 20,
        krange = range(1,19,1),
        adjust_class_weights = True,
        cdr3b_col = 'cdr3_b_aa',
        cdr3a_col = 'cdr3_a_aa',
        vb_col = 'v_b_gene',
        va_col = 'v_a_gene')

print(df)
```


## Alternative encoding work fine to data as integer vectorization 

Encode as any of the following:

```python
# Single chain Vgene only
e_va    = tg.encode_va_only(data)
e_vb    = tg.encode_vb_only(data)
# Single chain CDR3 only
e_cdr3b = tg.encode_cdr3b_only(data)
e_cdr3a = tg.encode_cdr3a_only(data)
# Single chain all CDRs
e_beta  = tg.encode_tcrs_b(data)
e_alpha = tg.encode_tcrs_a(data)    
# Paired chain all CDRs
e_full = tg.encode_tcrs(data) 

# compute small arrays for any type of encoding
arr_vb    = tg.compute_array(e_vb, e_vb)
arr_va    = tg.compute_array(e_va, e_va)
arr_cdr3b = tg.compute_array(e_cdr3b, e_cdr3b)
arr_cdr3a = tg.compute_array(e_cdr3a, e_cdr3a)
arr_beta  = tg.compute_array(e_beta , e_beta )
arr_alpha = tg.compute_array(e_alpha, e_alpha)
arr_full  = tg.compute_array(e_full, e_full)     
```


#### Package data
```python
import sys
import os
sys.path.append('/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/')
import tcrdistgpu
import importlib.util 
package_name = 'tcrdistgpu'
package_spec = importlib.util.find_spec('tcrdistgpu')
package_location = package_spec.origin
package_data = os.path.join(os.path.dirname(package_location), 'data')
print(os.listdir(package_data))
```











