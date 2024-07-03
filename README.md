# tcrdistgpu

## Credit

The idea and original code for a GPU accerlated TCRdist -- mikhail.pogorelyy@stjude.org (Mikhail Pogorelyy)

## Usage with CPU

dependencies: pandas, numpy

```python
from tcrdistgpu.distance import TCRgpu
import pandas as pd
tst_tcr = pd.read_csv("data/tmp_tcr.tsv", sep="\t").head(2000)
tg = TCRgpu(tcrs=tst_tcr, mode='cpu', kbest=10)
tg.encode_tcrs()
sorted_indices, sorted_smallest_k_values = tg.compute()
tg.sanity_test_nn_seqs(i=0, max_dist=150)
```

## Usage with GPU 

dependencies: cupy, pandas, numpy

```python
from tcrdistgpu.distance import TCRgpu
import pandas as pd
tst_tcr = pd.read_csv("data/tmp_tcr.tsv", sep="\t").head(2000)
tg = TCRgpu(tcrs=tst_tcr, mode='cuda', kbest=10)
tg.encode_tcrs()
sorted_indices, sorted_smallest_k_values = tg.compute()
tg.sanity_test_nn_seqs(i=0, max_dist=150)
```

See collab example with a T4 instance: https://colab.research.google.com/drive/1qIjuHblybT7RTFPaDO3ToYWTMPRLNNrW?usp=sharing


## Current status

Code works and is very fast compared to legacy tcrdist3, but it does not quite reproduce results 
from tcrdist3 particulary at large TCRdists. 

E.g., see tests/test_distance/test_x_compute_vs_dash which currenlty fails.

