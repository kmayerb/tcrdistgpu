# TCRdistGPU Step-by-Step Walkthrough: Finding Statistically Anomalous T-Cell Receptors

This walkthrough demonstrates how to use `tcrdistgpu` and `tcrshuffler` to identify T-cell receptors (TCRs) with statistically anomalous clustering patterns by comparing against shuffled background distributions.

## Overview

The key insight is that some TCRs cluster together more than expected by chance, suggesting they may recognize the same antigen. By comparing clustering patterns in real data versus shuffled backgrounds, we can identify these potentially antigen-specific groups.

## Prerequisites

Install the required packages:

```bash
# Core packages
pip install tcrdistgpu tcrshuffler fishersapi pandas numpy networkx

# For visualization (optional)
pip install matplotlib seaborn pygraphviz
```

## Step 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

# Core tcrdistgpu imports
from tcrdistgpu.distance import TCRgpu
from tcrdistgpu.radius import calc_radii, count_neighbors_below_radii_per_row, threshold_csr_rows
from tcrdistgpu.draw import draw_categories, draw_continuous

# Shuffling and statistical testing
from tcrshuffler.core import shuffle
from fishersapi import fishers_vec
```

## Step 2: Load and Examine Your Data

```python
# Load your TCR data (adjust path as needed)
data_file = 'path/to/your/tcr_data.tsv'
data = pd.read_csv(data_file, sep='\t')

print(f"Loaded {len(data)} TCRs")
print("Column names:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# Define your column names (adjust these to match your data)
v_col = 'vMaxResolved'     # V gene column
cdr3_col = 'aminoAcid'     # CDR3 sequence column
j_col = 'jMaxResolved'     # J gene column
chain = 'B'                # Which chain (A or B)
```

## Step 3: Data Cleaning and Validation

### 3.1: Remove Missing Values

```python
print(f"Starting with {len(data)} TCRs")

# Remove rows with missing gene information
data = data[data[v_col].notna()].reset_index(drop=True)
data = data[data[j_col].notna()].reset_index(drop=True)

print(f"After removing missing genes: {len(data)} TCRs")
```

### 3.2: Standardize Gene Names

```python
# Add allele notation if missing (tcrdistgpu expects *01 format)
data[v_col] = data[v_col].apply(lambda x: f"{x}*01" if x.find("*") == -1 else x)
data[j_col] = data[j_col].apply(lambda x: f"{x}*01" if x.find("*") == -1 else x)

print("Gene names standardized")
```

### 3.3: Filter for Recognized Genes

```python
# Initialize TCRgpu to get list of recognized genes
tg = TCRgpu()

# Keep only TCRs with recognized V genes
initial_count = len(data)
data = data[data[v_col].isin(tg.params_vec.keys())].reset_index(drop=True)

print(f"After filtering for recognized V genes: {len(data)} TCRs")
print(f"Removed {initial_count - len(data)} TCRs with unrecognized V genes")
```

### 3.4: Validate CDR3 Sequences

```python
# Define valid amino acids
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Function to check if CDR3 is valid
def is_valid_cdr3(cdr3):
    if not isinstance(cdr3, str):
        return False
    return all(aa in amino_acids for aa in cdr3)

# Apply validation
initial_count = len(data)
data = data[data[cdr3_col].apply(is_valid_cdr3)].reset_index(drop=True)

print(f"After CDR3 validation: {len(data)} TCRs")
print(f"Removed {initial_count - len(data)} TCRs with invalid CDR3 sequences")
```

### 3.5: Filter by CDR3 Length

```python
# Remove very short CDR3s (likely errors)
initial_count = len(data)
data = data[data[cdr3_col].apply(lambda x: len(x) > 5)].reset_index(drop=True)

print(f"After length filtering: {len(data)} TCRs")
print(f"Removed {initial_count - len(data)} TCRs with CDR3 length ≤ 5")

print(f"\nFinal cleaned dataset: {len(data)} TCRs")
```

## Step 4: Generate Shuffled Background

The shuffled background preserves gene usage frequencies but breaks up any antigen-specific clustering patterns.

```python
print("Generating shuffled background...")

# Create shuffled dataset
shuffled = shuffle(
    tcrs=data,
    chain=chain,
    v_col=v_col,
    cdr3_col=cdr3_col,
    j_col=j_col,
    depth=10,          # Controls how thoroughly sequences are shuffled
    random_seed=42,    # For reproducibility
    return_presuffled=False,
    return_errors=False
)

print(f"Generated {len(shuffled)} shuffled TCRs")
print(f"Original/shuffled ratio: {len(data)}/{len(shuffled)} = {len(data)/len(shuffled):.2f}")
```

## Step 5: Encode TCR Sequences

TCRdistGPU needs to encode the TCRs into numerical representations before computing distances.

```python
print("Encoding TCR sequences...")

# Initialize TCRgpu with your parameters
mode = 'cuda'  # Use 'cpu' if no GPU available
tg = TCRgpu(mode=mode, cdr3b_col=cdr3_col, vb_col=v_col, chunk_size=10)

# Encode original TCRs
print("Encoding original TCRs...")
encoding = tg.encode_tcrs_b(data)
print(f"Encoded {len(encoding)} original TCRs")

# Encode shuffled TCRs
print("Encoding shuffled TCRs...")
encoding_shuffled = tg.encode_tcrs_b(shuffled)
print(f"Encoded {len(encoding_shuffled)} shuffled TCRs")
```

## Step 6: Compute Distance Matrices

We need two distance matrices:
- **S**: Distances between original TCRs (to find clusters)
- **S2**: Distances from original TCRs to shuffled TCRs (for background comparison)

```python
print("Computing distance matrices...")

# Compute distances from original TCRs to shuffled background
print("Computing original-to-shuffled distances...")
S2 = tg.compute_csr(encoding, encoding_shuffled, max_k=20)
print(f"S2 matrix shape: {S2.shape}")

# Compute distances between original TCRs
print("Computing original-to-original distances...")
S = tg.compute_csr(encoding, encoding, max_k=20)
print(f"S matrix shape: {S.shape}")

print("Distance computation complete")
```

## Step 7: Calculate Statistical Radii

For each TCR, we determine a radius that captures the statistically expected number of neighbors based on the shuffled background.

```python
print("Calculating statistical radii...")

# Calculate radii based on shuffled background
alpha = 1e-5  # Significance level (very stringent)
radii = calc_radii(distance_matrix=S2, alpha=alpha)

# Apply upper limit to prevent extremely large radii
max_radius = 25  # Maximum TCR distance units
radii = np.array([min(x, max_radius) for x in radii])

print(f"Calculated radii for {len(radii)} TCRs")
print(f"Mean radius: {np.mean(radii):.2f}")
print(f"Median radius: {np.median(radii):.2f}")
print(f"Max radius: {np.max(radii):.2f}")
print(f"TCRs with max radius (capped): {np.sum(radii == max_radius)}")
```

## Step 8: Count Neighbors Within Radii

For each TCR, count how many neighbors it has within its statistical radius in both the original and shuffled data.

```python
print("Counting neighbors within radii...")

# Count neighbors in original data
nn_original = count_neighbors_below_radii_per_row(S, radii)

# Count neighbors in shuffled data  
nn_shuffled = count_neighbors_below_radii_per_row(S2, radii)

print(f"Neighbor counts calculated")
print(f"Mean neighbors in original data: {np.mean(nn_original):.2f}")
print(f"Mean neighbors in shuffled data: {np.mean(nn_shuffled):.2f}")
```

## Step 9: Prepare Results DataFrame

```python
# Create results dataframe
result = data.copy()
result['radii'] = radii
result['nn_below_radii'] = nn_original      # Neighbors in original data
result['nn_below_radii_sh'] = nn_shuffled   # Neighbors in shuffled data

print("Results dataframe prepared")
print(f"Shape: {result.shape}")
```

## Step 10: Statistical Testing with Fisher's Exact Test

We use Fisher's exact test to determine if each TCR has significantly more neighbors than expected by chance.

### 10.1: Set Up Contingency Tables

```python
# For each TCR, create a 2x2 contingency table:
#                   | Neighbors | Non-neighbors |
# Original data     |     a     |       c       |
# Shuffled data     |     b     |       d       |

result['a'] = result['nn_below_radii']                    # Neighbors in original
result['b'] = result['nn_below_radii_sh']                 # Neighbors in shuffled  
result['c'] = len(data) - result['a']                     # Non-neighbors in original
result['d'] = len(shuffled) - result['b']                 # Non-neighbors in shuffled

print("Contingency tables prepared")
```

### 10.2: Perform Fisher's Exact Tests

```python
print("Running Fisher's exact tests...")

# Perform vectorized Fisher's exact tests
odds_ratios, p_values = fishers_vec(result['a'], result['b'], result['c'], result['d'])

result['odds_ratio'] = odds_ratios
result['p_exact'] = p_values

print("Statistical testing complete")
```

## Step 11: Analyze Results

### 11.1: Summary Statistics

```python
print("\n" + "="*50)
print("STATISTICAL ANALYSIS RESULTS")
print("="*50)

# Overall statistics
print(f"Total TCRs analyzed: {len(result)}")
print(f"Mean p-value: {np.mean(result['p_exact']):.6f}")
print(f"Median p-value: {np.median(result['p_exact']):.6f}")

# Count significant TCRs at different thresholds
thresholds = [0.05, 0.01, 0.001, 0.0001, 0.00001]
for threshold in thresholds:
    count = (result['p_exact'] < threshold).sum()
    percentage = (count / len(result)) * 100
    print(f"TCRs with p < {threshold}: {count} ({percentage:.2f}%)")
```

### 11.2: Examine Top Significant TCRs

```python
# Focus on highly significant TCRs
significant_threshold = 0.0005
significant_tcrs = result.query(f'p_exact < {significant_threshold}')

print(f"\n{len(significant_tcrs)} TCRs with p < {significant_threshold}")

if len(significant_tcrs) > 0:
    # Sort by p-value and show top results
    top_tcrs = significant_tcrs.nsmallest(10, 'p_exact')
    
    print("\nTop 10 most significant TCRs:")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(top_tcrs.iterrows()):
        v_gene = row[v_col]
        cdr3 = row[cdr3_col]
        j_gene = row[j_col]
        p_val = row['p_exact']
        neighbors = row['nn_below_radii']
        odds = row['odds_ratio']
        
        print(f"{i+1:2d}. {v_gene} + {cdr3} + {j_gene}")
        print(f"    p-value: {p_val:.2e}, neighbors: {neighbors}, odds ratio: {odds:.2f}")
        print()
```

## Step 12: Create Thresholded Distance Matrix

Before creating the network, we need to threshold the distance matrix using our calculated radii to keep only the meaningful connections.

```python
from tcrdistgpu.radius import threshold_csr_rows

print("Thresholding distance matrix with calculated radii...")

# Apply radii thresholds to the distance matrix
# This keeps only distances below each TCR's statistical radius
Sx = threshold_csr_rows(S, radii)

print(f"Original matrix S shape: {S.shape}")
print(f"Thresholded matrix Sx shape: {Sx.shape}")
print(f"Original matrix non-zero elements: {S.nnz}")
print(f"Thresholded matrix non-zero elements: {Sx.nnz}")
print(f"Reduction in connections: {((S.nnz - Sx.nnz) / S.nnz * 100):.1f}%")
```

## Step 13: Extract Significant TCRs for Network Analysis

Now we'll create a network focusing only on the statistically significant TCRs.

### 13.1: Get Significant TCR Indices

```python
# Get indices of significant TCRs for network analysis
significant_threshold = 0.0005  # Adjust as needed
significant_tcrs = result.query(f'p_exact < {significant_threshold}')
ix = sorted(significant_tcrs.index.tolist())

print(f"Found {len(ix)} significant TCRs for network analysis")
print(f"Significant TCR indices: {ix[:10]}..." if len(ix) > 10 else f"Significant TCR indices: {ix}")
```

### 13.2: Create Submatrix with Only Significant TCRs

```python
# Note: keep_only_rows function may need to be implemented
# Here's a simple implementation to extract rows and columns for significant TCRs

def keep_only_rows(matrix, indices):
    """
    Extract submatrix containing only specified row/column indices
    """
    # Convert to dense if needed for indexing, then back to sparse
    if len(indices) == 0:
        return matrix[:0, :0]  # Return empty matrix
    
    # Extract submatrix using advanced indexing
    submatrix = matrix[indices, :][:, indices]
    return submatrix

if len(ix) > 1:
    print("Creating submatrix with significant TCRs only...")
    Sx2 = keep_only_rows(Sx, ix)
    
    print(f"Significant TCR submatrix shape: {Sx2.shape}")
    print(f"Non-zero elements in submatrix: {Sx2.nnz}")
else:
    print("Not enough significant TCRs for network analysis")
    Sx2 = None
```

## Step 14: Network Visualization

Now we can create and visualize the network of significant TCRs.

### 14.1: Create Network Graph

```python
if Sx2 is not None and Sx2.nnz > 0:
    print("Creating network from significant TCRs...")
    
    # Convert sparse matrix to NetworkX graph
    G = nx.from_scipy_sparse_array(Sx2)
    
    # Remove isolated nodes (TCRs with no connections)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    print(f"Network created:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Isolated nodes removed: {len(isolated_nodes)}")
    
    if G.number_of_nodes() == 0:
        print("Warning: No connected nodes in network")
        G = None
else:
    print("Cannot create network: no significant connections found")
    G = None
```

### 14.2: Prepare Data for Visualization

```python
if G is not None and G.number_of_nodes() > 0:
    # Create categorical variable for visualization
    # Map the original significant TCR data to the network nodes
    
    # Extract data for significant TCRs only
    sig_data = data.iloc[ix].copy()
    
    # Create size/color categories based on templates (if available)
    if 'templates' in sig_data.columns:
        sig_data['gt1'] = sig_data['templates'].apply(
            lambda x: 10 if x > 10 else 5 if x > 5 else 2 if x > 1 else 1
        )
        print("Created template-based categories (gt1)")
    else:
        # Create categories based on neighbor count if templates not available
        sig_data['gt1'] = result.iloc[ix]['nn_below_radii'].apply(
            lambda x: 10 if x > 10 else 5 if x > 5 else 2 if x > 1 else 1
        )
        print("Created neighbor-based categories (gt1)")
    
    print("Data prepared for visualization")
    print(f"Category distribution: {sig_data['gt1'].value_counts().sort_index()}")
```

### 14.3: Create Network Visualizations

```python
if G is not None and G.number_of_nodes() > 0:
    print("Creating network visualizations...")
    
    try:
        # Categorical visualization (colored by template/neighbor count)
        draw_categories(
            G=G, 
            data=sig_data, 
            color_col='gt1', 
            width=10, 
            height=10,
            size_col='gt1', 
            size_scale=1, 
            savefig='tcr_network_categories.png',
            legend=True
        )
        print("✓ Categorical network plot saved as 'tcr_network_categories.png'")
        
    except Exception as e:
        print(f"Categorical visualization failed: {e}")
    
    try:
        # Continuous visualization (colored by p-values)
        # Add p-values to the significant data
        sig_data_with_p = sig_data.copy()
        sig_data_with_p['nlog10_pexact'] = -np.log10(result.iloc[ix]['p_exact'])
        
        draw_continuous(
            G=G, 
            data=sig_data_with_p,
            continuous_color='nlog10_pexact',
            width=10, 
            height=10,
            size_col='gt1', 
            size_scale=1, 
            vmin=0, 
            vmax=12,
            savefig='tcr_network_pvalues.png'
        )
        print("✓ Continuous network plot saved as 'tcr_network_pvalues.png'")
        
    except Exception as e:
        print(f"Continuous visualization failed: {e}")

else:
    print("Skipping visualization: no network to display")
```

### 14.4: Network Statistics

```python
if G is not None and G.number_of_nodes() > 0:
    # Calculate basic network properties
    print("\nNetwork Statistics:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    
    if len(components) > 0:
        component_sizes = [len(comp) for comp in components]
        print(f"Largest component size: {max(component_sizes)}")
        print(f"Component size distribution: {sorted(component_sizes, reverse=True)[:10]}")
        
        # Show details of largest components
        print("\nLargest connected components:")
        for i, component in enumerate(sorted(components, key=len, reverse=True)[:3]):
            print(f"Component {i+1}: {len(component)} nodes")
            # Map back to original TCR indices
            original_indices = [ix[node] for node in component]
            component_tcrs = data.iloc[original_indices]
            print(f"  TCR examples: {component_tcrs[cdr3_col].head(3).tolist()}")
```

## Step 15: Save Results

```python
# Save the complete results
output_file = 'tcr_statistical_analysis_results.csv'
result.to_csv(output_file, index=False)
print(f"Complete results saved to: {output_file}")

# Save just the significant TCRs
if len(significant_tcrs) > 0:
    significant_file = 'significant_tcrs.csv'
    significant_tcrs.to_csv(significant_file, index=False)
    print(f"Significant TCRs saved to: {significant_file}")

# Create a summary report
summary_file = 'analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("TCRdistGPU Statistical Analysis Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset: {data_file}\n")
    f.write(f"Total TCRs analyzed: {len(result)}\n")
    f.write(f"Shuffled background size: {len(shuffled)}\n")
    f.write(f"Statistical significance threshold: {significant_threshold}\n\n")
    
    for threshold in [0.05, 0.01, 0.001, 0.0001, 0.00001]:
        count = (result['p_exact'] < threshold).sum()
        percentage = (count / len(result)) * 100
        f.write(f"TCRs with p < {threshold}: {count} ({percentage:.2f}%)\n")

print(f"Analysis summary saved to: {summary_file}")
```

## Step 16: Interpretation and Next Steps

```python
print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)

print("""
Key Results to Look For:

1. LOW P-VALUES (< 0.001):
   - These TCRs cluster more than expected by chance
   - May represent antigen-specific groups
   - Higher neighbor counts in original vs shuffled data

2. HIGH ODDS RATIOS (> 2.0):
   - Strong evidence of non-random clustering
   - TCR has many more neighbors than expected

3. NETWORK CLUSTERS:
   - Connected components in the network graph
   - Groups of similar TCRs that may recognize the same antigen

Next Steps:
- Examine the sequences of significant TCRs for common motifs
- Look up significant TCRs in databases (VDJdb, McPAS-TCR)
- Consider experimental validation of predicted antigen-specific groups
- Analyze the biological context (disease, treatment, etc.)
""")

# Show some example interpretations
if len(significant_tcrs) > 0:
    print(f"\nExample: Your most significant TCR has:")
    top_tcr = significant_tcrs.iloc[0]
    print(f"- p-value: {top_tcr['p_exact']:.2e} (highly significant)")
    print(f"- {top_tcr['nn_below_radii']} neighbors in original data")
    print(f"- {top_tcr['nn_below_radii_sh']} neighbors in shuffled data") 
    print(f"- Odds ratio: {top_tcr['odds_ratio']:.2f}")
    print(f"- This suggests non-random clustering, possibly antigen-specific")

print(f"\nAnalysis complete! Check your output files for detailed results.")
```

## Understanding the Method

This approach works by:

1. **Creating a null hypothesis**: Shuffled TCRs represent random clustering patterns
2. **Measuring clustering**: Count neighbors within statistically-determined radii  
3. **Statistical testing**: Compare real vs shuffled neighbor counts using Fisher's exact test
4. **Identifying anomalies**: TCRs with significantly more neighbors than expected by chance

The key insight is that antigen-specific TCRs should cluster together more than random TCRs, allowing us to identify potentially functional TCR groups in an unbiased way.

## Troubleshooting

**Common Issues:**

- **No significant TCRs found**: Try relaxing the p-value threshold or increasing shuffle depth
- **GPU memory errors**: Reduce chunk_size or use mode='cpu'  
- **Empty distance matrices**: Check that your gene names match tcrdistgpu's database
- **All radii at maximum**: Your data may be too diverse; consider focusing on specific subsets

**Performance Tips:**

- Use GPU mode for large datasets (>10,000 TCRs)
- Adjust chunk_size based on your GPU memory
- Consider subsampling very large datasets for exploratory analysis