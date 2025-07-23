from scipy.sparse import csr_matrix
import numpy as np

"""
Context:

In the development of tcrdist3 (eLife 202), 
a key concept was identifying a radius where the expected neighbor 
frequency falls below a threshold (1:N), helping flag TCRs with unexpectedly high number of 
neighbors or draw a boundary where similar TCRs may have been produced by recent antigenic 
selection. TCRs with high generation probability require a smaller radius, 
while rarer TCRs may still have surprising neighbors even at larger distances 
(e.g., <24-36). A similar approach is possible wiht tcrdistgpu, but no inverse weighting 
is implemented  meaning estimates will likely be imprecise unless using a large background.
"""

def count_neighbors(
    matrix: csr_matrix, min_value) -> np.ndarray:
    """
    Wrapper to count the number of neighbors per row in a CSR matrix below a threshold.
    Accepts either a single float threshold or a per-row array of thresholds.

    Parameters
    ----------
    matrix : csr_matrix
        Sparse matrix in CSR format.
    min_value : float or np.ndarray
        Single threshold value or 1D array of thresholds per row.

    Returns
    -------
    np.ndarray
        1D array of counts per row.
    """
    if isinstance(min_value, np.ndarray):
        return count_neighbors_below_radii_per_row(matrix, min_value)
    else:
        return count_neighbors_below_threshold_per_row(matrix, min_value)


def calc_radii(
    distance_matrix: csr_matrix,
    alpha: float = 1e-5,
    scan_values: list[int] = [1,4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
) -> np.ndarray:
    """
    Compute a radius per row (e.g., per clonotype) such that the expected neighbor frequency 
    under that radius is less than a specified threshold alpha.

    Parameters
    ----------
    distance_matrix : csr_matrix
        Sparse CSR matrix representing pairwise distances (e.g., TCRdist values).
    alpha : float, optional
        Threshold for the expected neighbor frequency (default: 2e-5).
    scan_values : list of int, optional
        List of radius thresholds to scan (default: [4, 8, ..., 48]).

    Returns
    -------
    np.ndarray
        1D array of radius values (from `scan_values`) per row where the neighbor frequency drops below `alpha`.
        If no radius satisfies the threshold, the result is undefined (likely returns smallest scan value).
    """
    n_rows, n_cols = distance_matrix.shape
    store = np.zeros((n_rows, len(scan_values)))

    for i, radius in enumerate(scan_values):
        row_counts = count_neighbors_below_threshold_per_row(distance_matrix, min_value=radius)
        store[:, i] = row_counts

    store /= n_cols  # Convert to neighbor frequency per row
    ix = largest_column_less_than_x(store, alpha)
    radius_per_clonotype = np.array(scan_values)[ix]

    return radius_per_clonotype




def count_neighbors_below_threshold_per_row(matrix: csr_matrix, min_value: float) -> np.ndarray:
    """
    Count the number of neighbors per row in a CSR matrix with values less than a specified threshold.

    Parameters
    ----------
    matrix : csr_matrix
        Sparse matrix in CSR format.
    min_value : float
        Threshold value to compare against.

    Returns
    -------
    np.ndarray
        1D array of shape (n_rows,) where each entry is the count of values < min_value in that row.
    """
    mask = matrix.data < min_value  # Boolean mask for values below threshold
    row_indices = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    counts = np.zeros(matrix.shape[0], dtype=int)
    np.add.at(counts, row_indices[mask], 1)
    return counts

def count_neighbors_below_radii_per_row(
    matrix: csr_matrix, min_values: np.ndarray
) -> np.ndarray:
    """
    Count the number of neighbors per row in a CSR matrix where values are less than a
    row-specific threshold, using a memory-efficient approach.

    Parameters
    ----------
    matrix : csr_matrix
        Sparse matrix in CSR format.
    min_values : np.ndarray
        1D array of shape (n_rows,) with threshold values for each row.

    Returns
    -------
    np.ndarray
        1D array of shape (n_rows,) with counts of values < threshold in each row.
    """
    if matrix.shape[0] != len(min_values):
        raise ValueError("Length of min_values must match number of rows in matrix.")

    counts = np.zeros(matrix.shape[0], dtype=int)
    for i in range(matrix.shape[0]):
        row_start, row_end = matrix.indptr[i], matrix.indptr[i + 1]
        row_data = matrix.data[row_start:row_end]
        counts[i] = np.count_nonzero(row_data < min_values[i])

    return counts

def largest_column_less_than_x(arr: np.ndarray, x: float) -> np.ndarray:
    """
    Find the largest column index in each row where the value is less than x.

    Parameters
    ----------
    arr : np.ndarray
        2D NumPy array with rows in ascending order.
    x : float
        Threshold value.

    Returns
    -------
    np.ndarray
        1D array of column indices for each row. Returns -1 if no value in the row is less than x.
    """
    mask = arr < x
    reversed_mask = mask[:, ::-1]
    last_true_idx_from_right = np.argmax(reversed_mask, axis=1)
    has_true = mask.any(axis=1)
    col_idx = mask.shape[1] - 1 - last_true_idx_from_right
    return np.where(has_true, col_idx, 0)


