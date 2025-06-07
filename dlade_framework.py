"""
DLADE: Dataset Landscape Density Evaluation Framework

This script implements the DLADE framework for evaluating density-based clustering
results. It processes various datasets, applies clustering algorithms (DBSCAN,
OPTICS, MeanShift) after hyperparameter tuning, and evaluates the outcomes using
a suite of novel density-aware metrics based on a Kernel Density
Estimate (KDE), alongside traditional internal validation indices.

The framework operates as follows:
1. Loads and preprocesses datasets.
2. For each dataset and each specified KDE configuration:
    a. Estimates a dataset-wide density function (psi).
    b. Calculates overall statistics (mean, std dev) of this psi.
    c. For each clustering algorithm (DBSCAN, OPTICS, MeanShift):
        i. Tunes hyperparameters using Silhouette score.
        ii. Runs the algorithm with best-tuned parameters.
        iii. Evaluates the resulting clustering using:
            - DLADE metrics (OverallClust_AvgDens_Z, Sep_OverallMinPath_Z, DDSI_psi, etc.)
            - Baseline metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, DBCV).
        iv. Saves plots and detailed results to CSV files.
3. Summarizes the best PSI estimator for each dataset based on average DDSI_psi.
4. Prepares and runs a demo section illustrating DLADE metrics for the best
   clustering run (highest DDSI_psi) on each dataset.

To run this script:
1. Ensure all dependencies are installed (see requirements.txt or README.md).
2. Place the necessary dataset files in a 'datasets/' subdirectory.
3. Place 'DBCV.py' in the same directory as this script or ensure it's in PYTHONPATH.
4. Execute the script from its directory: `python dlade_framework.py`
   Results (CSVs and plots) will be saved in a 'results/dlade_results_7datasets/' subdirectory.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys # Keep for sys.path manipulation if DBCV.py is elsewhere or for CLI args later
import ast

# --- Local Path Configuration ---
# Assumes the script is run from the root of the project repository
# and datasets are in a 'datasets' subdirectory.
# Results will be saved in a 'results' subdirectory.

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results")
RESULTS_SUBDIR_NAME = "dlade_results_7datasets"
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, RESULTS_SUBDIR_NAME)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Create if it doesn't exist

# --- Definitions for KDE and Algos ---
KDE_CONFIGURATIONS = [
    {'kernel': 'gaussian', 'bandwidth_method': 'silverman', 'name': 'KDE_Gauss_Silverman'},
    {'kernel': 'epanechnikov', 'bandwidth_method': 'silverman', 'name': 'KDE_Epanech_Silverman'},
    {'kernel': 'gaussian', 'bandwidth_method': 'cv', 'name': 'KDE_Gauss_CV'},
    {'kernel': 'epanechnikov', 'bandwidth_method': 'cv', 'name': 'KDE_Epanech_CV'},
]

ALGO_DEFINITIONS = {"DBSCAN": DBSCAN, "OPTICS": OPTICS, "MeanShift": MeanShift}


# --- Helper and Framework Functions ---
def plot_clusters_and_kde(data, labels, kde_psi_estimator_2d, dataset_name,
                          algo_name_for_labels, psi_estimator_name_text="",
                          clustering_params_text=""):
    """
    Generates and saves a 2-panel plot for 2D clustering results, showing:
    1. Data points colored by cluster label with KDE contours.
    2. KDE map.

    Skips plotting if data is not 2D.

    Args:
        data (np.ndarray): The 2D data points (scaled).
        labels (np.ndarray): Cluster labels for each data point.
        kde_psi_estimator_2d (sklearn.neighbors.KernelDensity): Fitted 2D KDE estimator for psi.
        dataset_name (str): Name of the dataset (used in title and filename).
        algo_name_for_labels (str): Name of the clustering algorithm (used in title and filename).
        psi_estimator_name_text (str, optional): Name of the psi estimator (for plot title).
        clustering_params_text (str, optional): Text describing clustering parameters (for plot annotation).
    """
    if data.shape[1] != 2:
        print(f"Plotting skipped for {dataset_name} - {algo_name_for_labels}: Data is not 2D (shape: {data.shape})")
        return

    clean_dataset_name_for_file = dataset_name.replace(' (2D Proj.)', '')

    if not hasattr(kde_psi_estimator_2d, 'tree_') or kde_psi_estimator_2d.tree_.data.shape[1] != 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        fig.suptitle(f"Dataset: {dataset_name} - Clustered by: {algo_name_for_labels} (KDE Plot Error for {psi_estimator_name_text})", fontsize=14)
        unique_labels_plot_error = np.unique(labels)
        num_distinct_labels_error = len(unique_labels_plot_error)
        palette_error = sns.color_palette("husl", n_colors=max(1, num_distinct_labels_error))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette_error, ax=ax, edgecolor='black', s=30, legend='full')
        ax.set_title(f'Data Points (Labels: {algo_name_for_labels}) - KDE Map Failed')
        if clustering_params_text:
            ax.text(0.95, 0.05, clustering_params_text, transform=ax.transAxes, fontsize=8, va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))
    else:
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        log_density_grid = kde_psi_estimator_2d.score_samples(grid_points)
        zz = np.exp(log_density_grid).reshape(xx.shape)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        title_text = f"Dataset: {dataset_name} - Clustered by: {algo_name_for_labels}"
        if psi_estimator_name_text:
            title_text += f" (Psi: {psi_estimator_name_text})"
        fig.suptitle(title_text, fontsize=14)

        axes[0].contourf(xx, yy, zz, cmap='viridis', alpha=0.6, levels=15)
        unique_labels_plot = np.unique(labels)
        num_distinct_labels = len(unique_labels_plot)
        palette = sns.color_palette("husl", n_colors=max(1, num_distinct_labels))

        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette, ax=axes[0], edgecolor='black', s=30, legend='full')
        axes[0].set_title('Data Points & KDE Map')
        if clustering_params_text:
            axes[0].text(0.95, 0.05, clustering_params_text, transform=axes[0].transAxes, fontsize=8, va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))

        axes[1].contourf(xx, yy, zz, cmap='viridis', levels=15, alpha=0.8)
        axes[1].contour(xx, yy, zz, colors='k', linewidths=0.5, levels=15)
        axes[1].set_title(f'Dataset-wide KDE Map ({psi_estimator_name_text if psi_estimator_name_text else "Default Psi"})')

        for ax_curr in axes:
            ax_curr.set_xlim(x_min, x_max)
            ax_curr.set_ylim(y_min, y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename_base = f"{clean_dataset_name_for_file}_{algo_name_for_labels}"
    if psi_estimator_name_text:
        plot_filename_base += f"_Psi_{psi_estimator_name_text}"
    plot_filename_base += "_tuned_plot.png"

    plt.savefig(os.path.join(RESULTS_DIR, plot_filename_base))
    print(f"Saved KDE plot to {os.path.join(RESULTS_DIR, plot_filename_base)}")
    plt.close(fig) # Close the figure to free memory


def load_dataset(filename, data_directory, delimiter='\t', skip_header=0, label_col=-1, needs_label_mapping=None):
    """
    Loads a dataset from a file.

    Args:
        filename (str): Name of the dataset file.
        data_directory (str): Path to the directory containing dataset files.
        delimiter (str, optional): Delimiter used in the file. Defaults to '\t'.
        skip_header (int, optional): Number of header rows to skip. Defaults to 0.
        label_col (int or None, optional): Index of the label column.
                                           -1 for last column, None if no labels. Defaults to -1.
        needs_label_mapping (dict, optional): Dictionary to map string labels to numeric. Defaults to None.

    Returns:
        tuple: (data_features, data_labels)
               - data_features (np.ndarray or None): Numeric feature array.
               - data_labels (np.ndarray or None): Labels array, or None if no labels.
               Returns (None, None) if loading fails or data is empty.
    """
    filepath = os.path.join(data_directory, filename)
    data, labels_true = None, None
    try:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, header=None, skiprows=skip_header)
        except pd.errors.ParserError:
            df = pd.read_csv(filepath, delimiter='\s+', header=None, skiprows=skip_header, engine='python')
        except FileNotFoundError:
            print(f"Error: File {filename} not found at {filepath}.")
            return None, None
        except Exception as e:
            print(f"Pandas error loading {filename}: {e}")
            return None, None

        if df.empty:
            print(f"Warning: Empty DataFrame for {filename}.")
            return None, None

        df.dropna(how='all', inplace=True)
        if df.empty:
            print(f"Warning: DataFrame became empty after dropping all-NaN rows for {filename}.")
            return None, None

        if label_col is None:
            data_df_features = df.apply(pd.to_numeric, errors='coerce')
            nan_rows = data_df_features.isnull().any(axis=1)
            if nan_rows.any():
                print(f"Warning: Found non-numeric values in feature columns of {filename} (no label_col). Dropping {nan_rows.sum()} rows.")
                data_df_features = data_df_features[~nan_rows]
            if data_df_features.empty:
                print(f"Warning: Feature data became empty after handling non-numeric values for {filename}.")
                return None, None
            data = data_df_features.values.astype(np.float64)
            labels_true = None
        else:
            num_cols = df.shape[1]
            actual_idx = label_col if label_col >= 0 else num_cols + label_col

            if not (0 <= actual_idx < num_cols):
                print(f"Warning: Label column index {label_col} (actual: {actual_idx}) out of bounds for {filename} with {num_cols} columns. Treating all as numeric features.")
                data_df_features = df.apply(pd.to_numeric, errors='coerce')
                nan_rows = data_df_features.isnull().any(axis=1)
                if nan_rows.any():
                    print(f"Warning: Found non-numeric values in feature columns of {filename}. Dropping {nan_rows.sum()} rows.")
                    data_df_features = data_df_features[~nan_rows]
                if data_df_features.empty:
                    print(f"Warning: Feature data became empty after handling non-numeric values for {filename}.")
                    return None, None
                data = data_df_features.values.astype(np.float64)
                labels_true = None
            else:
                feat_idx = [i for i in range(num_cols) if i != actual_idx]
                if not feat_idx:
                    print(f"Warning: No feature columns found for {filename} after selecting label column {actual_idx}.")
                    data = None
                    labels_true_series = df.iloc[:, actual_idx]
                else:
                    data_df_features = df.iloc[:, feat_idx].apply(pd.to_numeric, errors='coerce')
                    labels_true_series = df.iloc[:, actual_idx]

                    nan_rows_in_features = data_df_features.isnull().any(axis=1)
                    if nan_rows_in_features.any():
                        print(f"Warning: Found non-numeric feature values in {filename}. Dropping {nan_rows_in_features.sum()} rows.")
                        data_df_features = data_df_features[~nan_rows_in_features]
                        labels_true_series = labels_true_series[~nan_rows_in_features]

                    if data_df_features.empty:
                        print(f"Warning: Feature data became empty after handling non-numeric values for {filename}.")
                        return None, None
                    data = data_df_features.values.astype(np.float64)

                if needs_label_mapping and labels_true_series is not None:
                    mapped_labels = labels_true_series.map(needs_label_mapping)
                    if mapped_labels.isnull().any() and not labels_true_series[mapped_labels.isnull()].isnull().all():
                         unmapped_values = labels_true_series[mapped_labels.isnull() & labels_true_series.notnull()].unique()
                         if len(unmapped_values) > 0:
                            print(f"Warning: Label mapping for {filename} resulted in NaN values for originally non-NaN labels. Unmapped values: {unmapped_values}.")
                    labels_true = mapped_labels.values
                elif labels_true_series is not None:
                    labels_true = labels_true_series.values
                else:
                    labels_true = None


                if labels_true is not None:
                    temp_labels_series = pd.Series(labels_true)
                    # Attempt to convert to int if possible (e.g. numeric labels like 0.0, 1.0)
                    if pd.api.types.is_numeric_dtype(temp_labels_series) and temp_labels_series.notna().all() and np.isfinite(temp_labels_series).all():
                        try:
                            labels_true = temp_labels_series.astype(int).values
                        except ValueError:
                             pass # Keep as float or object if direct int conversion fails
                    elif temp_labels_series.isna().any():
                        print(f"Warning: Labels for {filename} contain NaN values after processing. Ensure this is expected.")

        num_objects = data.shape[0] if data is not None else 0
        num_features = data.shape[1] if data is not None else 0
        print(f"Loaded {filename}: {num_objects} objects, {num_features} features.")
        if num_objects == 0 and data is not None :
            print(f"Critical Warning: Dataset {filename} resulted in 0 objects after processing.")
            return None, None
        return data, labels_true

    except Exception as e:
        print(f"General error loading or processing {filename}: {e}")
        return None, None


def estimate_kde_bandwidth_silverman(data):
    """
    Estimates KDE bandwidth using Silverman's rule of thumb.

    Args:
        data (np.ndarray): Input data array (n_samples, n_features).

    Returns:
        float: Estimated bandwidth.
    """
    n, d_raw = data.shape
    if n == 0: return 0.1
    d = d_raw if d_raw > 0 else 1
    data_for_std = data if d > 0 else data.reshape(-1, 1)
    if d == 1 and data_for_std.ndim == 1:
        data_for_std = data_for_std.reshape(-1,1)

    sigma_vec = np.std(data_for_std, axis=0)
    if not isinstance(sigma_vec, np.ndarray):
        sigma_vec = np.array([sigma_vec])
    sigma_vec[sigma_vec < 1e-6] = 1e-6
    sigma = np.mean(sigma_vec)
    if sigma < 1e-6 : sigma = 1e-6

    bandwidth = (4 / (d + 2))**(1 / (d + 4)) * sigma * n**(-1 / (d + 4))
    return max(bandwidth, 1e-6)


def estimate_kde_bandwidth_cv(data, kernel_type='gaussian'):
    """
    Estimates KDE bandwidth using GridSearchCV with cross-validation.

    Args:
        data (np.ndarray): Input data array.
        kernel_type (str, optional): Type of kernel for KDE. Defaults to 'gaussian'.

    Returns:
        float: Estimated bandwidth. Falls back to Silverman's if CV fails or data is too small.
    """
    print(f"CV for KDE bw (k: {kernel_type})...", end=" ")
    n_total = len(data)
    if n_total < 10:
        print("Data too small, using Silverman's.")
        return estimate_kde_bandwidth_silverman(data)

    n_cv = min(n_total, 200) # Use a subset for CV if data is large
    if n_total <= n_cv:
        data_cv = data
    else:
        data_cv = data[np.random.choice(n_total, n_cv, replace=False)]

    cv_folds = min(3, len(data_cv)-1 if len(data_cv)>1 else 1)
    if cv_folds < 2:
        print("Not enough samples for CV folds, using Silverman's.")
        return estimate_kde_bandwidth_silverman(data)

    params = {'bandwidth': np.logspace(-1.5, 0.5, 3)} # Reduced grid for speed
    grid = GridSearchCV(KernelDensity(kernel=kernel_type), params, cv=cv_folds, n_jobs=-1)
    try:
        grid.fit(data_cv)
        print(f"Best CV bw: {grid.best_estimator_.bandwidth:.4f}")
        return grid.best_estimator_.bandwidth
    except Exception as e:
        print(f"CV KDE failed: {e}, falling back to Silverman's.")
        return estimate_kde_bandwidth_silverman(data)


def create_kde_estimator(data_scaled, kernel_type, bandwidth_method):
    """
    Creates and fits a KernelDensity estimator.

    Args:
        data_scaled (np.ndarray): Scaled input data.
        kernel_type (str): Kernel to use ('gaussian', 'epanechnikov', etc.).
        bandwidth_method (str): Method to estimate bandwidth ('silverman' or 'cv').

    Returns:
        sklearn.neighbors.KernelDensity: Fitted KDE estimator.
    """
    if bandwidth_method == 'cv':
        bw = estimate_kde_bandwidth_cv(data_scaled, kernel_type)
    else: # Default to Silverman
        bw = estimate_kde_bandwidth_silverman(data_scaled)
    print(f"KDE: k={kernel_type}, bw_meth={bandwidth_method}, selected_bw={bw:.4f}")
    estimator = KernelDensity(kernel=kernel_type, bandwidth=bw)
    estimator.fit(data_scaled)
    return estimator

def calculate_dataset_density_stats(data_O, kde_est):
    """
    Calculates mean and std dev of density values for the entire dataset.

    Args:
        data_O (np.ndarray): Original (or scaled) dataset.
        kde_est (sklearn.neighbors.KernelDensity): Fitted KDE estimator for psi.

    Returns:
        tuple: (mean_psi_O, std_psi_O, psi_O_values)
               - mean_psi_O (float): Mean of psi values over data_O.
               - std_psi_O (float): Standard deviation of psi values over data_O.
               - psi_O_values (np.ndarray): Array of psi values for each point in data_O.
    """
    log_psi_O = kde_est.score_samples(data_O)
    psi_O_values = np.exp(log_psi_O)
    return np.mean(psi_O_values), np.std(psi_O_values), psi_O_values

def z_score_density(val, mean_psi, std_psi):
    """
    Calculates the z-score of a density value.

    Args:
        val (float or None): The density value to normalize.
        mean_psi (float or None): The mean of the reference density distribution.
        std_psi (float or None): The std dev of the reference density distribution.

    Returns:
        float or np.nan: Z-score, or NaN if inputs are invalid or std_psi is too small.
    """
    if val is None or mean_psi is None or std_psi is None or \
       not np.isfinite(val) or not np.isfinite(mean_psi) or not np.isfinite(std_psi):
        return np.nan
    return (val - mean_psi) / std_psi if std_psi > 1e-9 else 0.0


def evaluate_clustering_framework(data_O, labels, kde_psi_estimator,
                                  mu_O_psi, sigma_O_psi, psi_O_values,
                                  n_separation_samples=5, n_path_points=30):
    """
    Evaluates a clustering result using the DLADE framework metrics.

    Args:
        data_O (np.ndarray): The original (or scaled) dataset.
        labels (np.ndarray): Cluster labels for each point in data_O.
        kde_psi_estimator (sklearn.neighbors.KernelDensity): Fitted KDE for psi.
        mu_O_psi (float): Mean density of the entire dataset.
        sigma_O_psi (float): Std dev of density of the entire dataset.
        psi_O_values (np.ndarray): Psi values for each point in data_O.
        n_separation_samples (int): Number of point pairs to sample for path density.
        n_path_points (int): Number of points along each path for density evaluation.

    Returns:
        dict: Dictionary of DLADE evaluation metrics.
    """
    
    results = {}
    unique_labels = np.unique(labels)
    cluster_labels_for_eval = [l for l in unique_labels if l >= 0]

    cluster_psi_values_dict = {l: psi_O_values[labels == l] for l in cluster_labels_for_eval if np.sum(labels == l) > 0}
    outlier_psi_values = psi_O_values[labels == -1]

    avg_cluster_densities_psi_list = []
    total_clustered_points = 0
    weighted_sum_sq_dev_dens_psi = 0

    for label_val in cluster_labels_for_eval:
        psi_in_cluster = cluster_psi_values_dict.get(label_val)
        if psi_in_cluster is None or len(psi_in_cluster) == 0:
            results[f'C_{label_val}_AvgDens_psi'], results[f'C_{label_val}_AvgDens_Z'], results[f'C_{label_val}_DensVar_psi'] = np.nan, np.nan, np.nan
            continue
        mu_C_psi, sigma_C_psi = np.mean(psi_in_cluster), np.std(psi_in_cluster)
        avg_cluster_densities_psi_list.append(mu_C_psi)
        results[f'C_{label_val}_AvgDens_psi'] = mu_C_psi
        results[f'C_{label_val}_AvgDens_Z'] = z_score_density(mu_C_psi, mu_O_psi, sigma_O_psi)
        results[f'C_{label_val}_DensVar_psi'] = sigma_C_psi

        total_clustered_points += len(psi_in_cluster)
        weighted_sum_sq_dev_dens_psi += np.sum((psi_in_cluster - mu_C_psi)**2)


    psi_CX_values = np.concatenate(list(cluster_psi_values_dict.values())) if cluster_psi_values_dict else np.array([])
    if len(psi_CX_values) > 0:
        mu_CX_psi, sigma_CX_psi_overall = np.mean(psi_CX_values), np.std(psi_CX_values)
        results['OverallClust_AvgDens_psi'] = mu_CX_psi
        results['OverallClust_AvgDens_Z'] = z_score_density(mu_CX_psi, mu_O_psi, sigma_O_psi)
        results['OverallClust_DensVar_psi'] = sigma_CX_psi_overall

        if total_clustered_points > 0:
            results['OverallClust_WeightedDensVar_psi'] = np.sqrt(weighted_sum_sq_dev_dens_psi / total_clustered_points)
        else:
            results['OverallClust_WeightedDensVar_psi'] = np.nan

        if len(avg_cluster_densities_psi_list) > 1: results['InterCluster_AvgDensVar_psi'] = np.std(avg_cluster_densities_psi_list)
        else: results['InterCluster_AvgDensVar_psi'] = np.nan
    else:
        results.update({'OverallClust_AvgDens_psi': np.nan, 'OverallClust_AvgDens_Z': np.nan,
                        'OverallClust_DensVar_psi': np.nan, 'InterCluster_AvgDensVar_psi': np.nan,
                        'OverallClust_WeightedDensVar_psi': np.nan})

    if len(outlier_psi_values) > 0:
        mu_Out_psi = np.mean(outlier_psi_values)
        results['Outlier_AvgDens_psi'], results['Outlier_AvgDens_Z'] = mu_Out_psi, z_score_density(mu_Out_psi, mu_O_psi, sigma_O_psi)
    else: results['Outlier_AvgDens_psi'], results['Outlier_AvgDens_Z'] = np.nan, np.nan

    cluster_coords_dict = {l: data_O[labels == l] for l in cluster_labels_for_eval if np.sum(labels == l) > 0}
    if len(cluster_coords_dict) >= 2:
        all_pairwise_min_path_densities = []
        cluster_ids_list = sorted(list(cluster_coords_dict.keys()))
        for i in range(len(cluster_ids_list)):
            for j in range(i + 1, len(cluster_ids_list)):
                cl_id_i, cl_id_j = cluster_ids_list[i], cluster_ids_list[j]
                c1_pts, c2_pts = cluster_coords_dict.get(cl_id_i), cluster_coords_dict.get(cl_id_j)
                if c1_pts is None or c2_pts is None or len(c1_pts) < 1 or len(c2_pts) < 1: continue
                current_pair_min_dens = []
                n_s1_actual = min(n_separation_samples, len(c1_pts))
                n_s2_actual = min(n_separation_samples, len(c2_pts))

                idx1 = np.random.choice(len(c1_pts), size=n_s1_actual, replace=(len(c1_pts) < n_s1_actual)) if n_s1_actual > 0 else []
                idx2 = np.random.choice(len(c2_pts), size=n_s2_actual, replace=(len(c2_pts) < n_s2_actual)) if n_s2_actual > 0 else []

                for u_i in idx1:
                    for v_i in idx2:
                        u, v = c1_pts[u_i], c2_pts[v_i]
                        if np.all(u==v): continue
                        lp = np.linspace(u,v,num=n_path_points)
                        try: current_pair_min_dens.append(np.min(np.exp(kde_psi_estimator.score_samples(lp))))
                        except Exception: current_pair_min_dens.append(np.inf)
                if current_pair_min_dens:
                    all_pairwise_min_path_densities.append(np.min(current_pair_min_dens))

        valid_paths = [d for d in all_pairwise_min_path_densities if np.isfinite(d) and d != np.inf]
        if valid_paths:
            min_min_path_psi = np.min(valid_paths)
            results['Sep_OverallMinPath_psi']= min_min_path_psi 
            results['Sep_OverallMinPath_Z']=z_score_density(min_min_path_psi, mu_O_psi, sigma_O_psi) 
        else: results.update({'Sep_OverallMinPath_psi': np.nan, 'Sep_OverallMinPath_Z': np.nan}) 
    else: results.update({'Sep_OverallMinPath_psi': np.nan, 'Sep_OverallMinPath_Z': np.nan}) 
    return results


def find_best_clustering_params(data_scaled, algo_class, param_grid,
                                primary_tuning_metric_config,
                                current_kde_psi_estimator_for_ddsi=None,
                                mu_O_psi_for_ddsi=None,
                                sigma_O_psi_for_ddsi=None,
                                psi_O_values_for_ddsi=None):
    """
    Finds the best hyperparameters for a given clustering algorithm using a specified metric.

    Args:
        data_scaled (np.ndarray): Scaled input data.
        algo_class (class): The clustering algorithm class (e.g., DBSCAN).
        param_grid (dict): Dictionary of parameters to search.
        primary_tuning_metric_config (dict): Configuration for the tuning metric.
            Includes 'name', 'func' (callable), and 'higher_is_better' (bool).
        current_kde_psi_estimator_for_ddsi (sklearn.neighbors.KernelDensity, optional):
            KDE estimator if tuning with DDSI_psi.
        mu_O_psi_for_ddsi (float, optional): Mean dataset density if tuning with DDSI_psi.
        sigma_O_psi_for_ddsi (float, optional): Std dev of dataset density if tuning with DDSI_psi.
        psi_O_values_for_ddsi (np.ndarray, optional): Psi values if tuning with DDSI_psi.

    Returns:
        tuple: (best_params, best_score)
               - best_params (dict or None): Best hyperparameter set found.
               - best_score (float): Score achieved by the best parameters.
    """
   
    metric_name = primary_tuning_metric_config["name"]
    metric_func = primary_tuning_metric_config["func"]
    higher_is_better = primary_tuning_metric_config["higher_is_better"]
    print(f"Tuning {algo_class.__name__} using {metric_name} as scorer...", end=" ")
    best_score = -np.inf if higher_is_better else np.inf
    best_params = None

    try: fallback_params = next(iter(ParameterGrid(param_grid)))
    except StopIteration: fallback_params = {}

    grid = ParameterGrid(param_grid)

    for params_iter in grid:
        current_params = params_iter.copy()
        try:
            if algo_class == MeanShift and current_params.get("bandwidth") is None:
                 n_s_bw = min(len(data_scaled), 500)
                 ms_bw = 0.1 if n_s_bw < 2 else estimate_bandwidth(data_scaled, quantile=0.2, n_samples=n_s_bw, n_jobs=-1)
                 current_params["bandwidth"] = max(ms_bw, 1e-3) if ms_bw is not None and np.isfinite(ms_bw) else 1.0

            model = algo_class(**current_params); labels = model.fit_predict(data_scaled)
            n_clusters_f = len(np.unique(labels[(labels != -1) & (labels != -2)]))
            score = np.nan
            if metric_name == "DDSI_psi":
                if n_clusters_f == 0: score = -np.inf
                else:
                    if not all(arg is not None for arg in [current_kde_psi_estimator_for_ddsi, mu_O_psi_for_ddsi, sigma_O_psi_for_ddsi, psi_O_values_for_ddsi]): score = np.nan
                    else:
                        fw_scores = evaluate_clustering_framework(data_scaled, labels, current_kde_psi_estimator_for_ddsi, mu_O_psi_for_ddsi, sigma_O_psi_for_ddsi, psi_O_values_for_ddsi)
                        ov_z, s_z = fw_scores.get('OverallClust_AvgDens_Z'), fw_scores.get('Sep_OverallMinPath_Z') 
                        if ov_z is not None and np.isfinite(ov_z):
                            if n_clusters_f >= 2 and s_z is not None and np.isfinite(s_z): score = ov_z - s_z
                            elif n_clusters_f == 1: score = ov_z
                            else: score = np.nan
                        else: score = np.nan
            else:
                if n_clusters_f < 2: score = -np.inf if higher_is_better else np.inf
                else: score = metric_func(data_scaled, labels)

            if not np.isfinite(score): score = -np.inf if higher_is_better else np.inf

            if (higher_is_better and score > best_score) or \
               (not higher_is_better and score < best_score):
                best_score, best_params = score, params_iter.copy()
        except Exception: continue

    if best_params is not None: print(f"Best params: {best_params} (Score: {best_score:.4f})")
    else:
        print(f"No successful run for tuning. Using fallback: {fallback_params}"); best_params = fallback_params
        if (not np.isfinite(best_score) or best_params is None) and fallback_params: # best_params might be None if grid is empty
            try:
                fb_params_to_run = fallback_params.copy()
                if algo_class == MeanShift and fb_params_to_run.get("bandwidth") is None:
                    n_s_bw_fb = min(len(data_scaled), 500)
                    ms_bw_fb = 0.1 if n_s_bw_fb < 2 else estimate_bandwidth(data_scaled, quantile=0.2, n_samples=n_s_bw_fb, n_jobs=-1)
                    fb_params_to_run["bandwidth"] = max(ms_bw_fb, 1e-3) if ms_bw_fb is not None and np.isfinite(ms_bw_fb) else 1.0

                fb_model = algo_class(**fb_params_to_run); fb_labels = fb_model.fit_predict(data_scaled)
                fb_n_clust = len(np.unique(fb_labels[(fb_labels!=-1)&(fb_labels!=-2)]))

                if metric_name == "DDSI_psi":
                    if fb_n_clust == 0: best_score = -np.inf
                    elif all(arg is not None for arg in [current_kde_psi_estimator_for_ddsi, mu_O_psi_for_ddsi, sigma_O_psi_for_ddsi, psi_O_values_for_ddsi]):
                        fw_s = evaluate_clustering_framework(data_scaled, fb_labels, current_kde_psi_estimator_for_ddsi, mu_O_psi_for_ddsi, sigma_O_psi_for_ddsi, psi_O_values_for_ddsi)
                        ov, sp = fw_s.get('OverallClust_AvgDens_Z'), fw_s.get('Sep_OverallMinPath_Z') 
                        if ov is not None and np.isfinite(ov):
                            if fb_n_clust >=2 and sp is not None and np.isfinite(sp) : best_score = ov - sp
                            elif fb_n_clust == 1: best_score = ov
                            else: best_score = np.nan
                        else: best_score = np.nan
                    else: best_score = np.nan
                else:
                    if fb_n_clust < 2: best_score = -np.inf if higher_is_better else np.inf
                    else: best_score = metric_func(data_scaled, fb_labels)
                if not np.isfinite(best_score): best_score = -np.inf if higher_is_better else np.inf
            except: best_score = -np.inf if higher_is_better else np.inf
    return best_params, best_score


# --- Main Experiment Function ---
def run_experiment(dataset_name, data_raw,
                   fixed_params_for_algos_in_dataset,
                   param_grids_for_algo_tuning,
                   primary_tuning_metric_config):
    """
    Runs the full experiment for a single dataset.
    This includes KDE estimation, algorithm tuning, clustering, and evaluation.

    Args:
        dataset_name (str): Name of the dataset.
        data_raw (np.ndarray): Raw feature data.
        fixed_params_for_algos_in_dataset (dict): Fixed parameters for algorithms
                                                 if no tuning grid is provided.
        param_grids_for_algo_tuning (dict): Parameter grids for tuning each algorithm.
        primary_tuning_metric_config (dict): Configuration for the primary tuning metric.

    Returns:
        dict: A dictionary containing all results, structured by psi_estimator_name
              and then by algorithm_name.
    """

    print(f"\n--- Running Experiment on Dataset: {dataset_name} ---")
    start_time_total = time.time()
    scaler = StandardScaler(); data_scaled = scaler.fit_transform(data_raw)

    master_results_for_dataset_by_psi = {}

    for kde_config in KDE_CONFIGURATIONS:
        psi_estimator_name = kde_config['name']
        print(f"\n\n=== Using Psi Estimator: {psi_estimator_name} for Dataset: {dataset_name} ===")
        current_kde_psi_estimator = create_kde_estimator(data_scaled, kde_config['kernel'], kde_config['bandwidth_method'])
        mu_O_psi, sigma_O_psi, psi_O_values = calculate_dataset_density_stats(data_scaled, current_kde_psi_estimator) 
        print(f"Dataset-wide stats for {psi_estimator_name}: mu_O_psi={mu_O_psi:.4f}, sigma_O_psi={sigma_O_psi:.4f}")

        results_for_this_psi_estimator = {}

        for algo_name, algo_class in ALGO_DEFINITIONS.items():
            print(f"\n-- Processing Algorithm: {algo_name} (Psi: {psi_estimator_name}) --")

            params_to_run_with = fixed_params_for_algos_in_dataset.get(algo_name, {}).copy()
            tuning_score_achieved = np.nan
            tuning_metric_name_display = "Fixed/Default"

            algo_param_grid = param_grids_for_algo_tuning.get(algo_name, {})
            if algo_param_grid:
                tuning_metric_name_display = primary_tuning_metric_config['name']

                is_ddsi_tuning = primary_tuning_metric_config["name"] == "DDSI_psi"
                tuned_params, score_from_tuning = find_best_clustering_params(
                    data_scaled, algo_class, algo_param_grid,
                    primary_tuning_metric_config,
                    current_kde_psi_estimator_for_ddsi=current_kde_psi_estimator if is_ddsi_tuning else None,
                    mu_O_psi_for_ddsi=mu_O_psi if is_ddsi_tuning else None,
                    sigma_O_psi_for_ddsi=sigma_O_psi if is_ddsi_tuning else None,
                    psi_O_values_for_ddsi=psi_O_values if is_ddsi_tuning else None
                )
                if tuned_params: params_to_run_with = tuned_params; tuning_score_achieved = score_from_tuning
                else: print(f"  Tuning with {tuning_metric_name_display} failed for {algo_name}, using fixed params: {params_to_run_with}")
            else:
                 print(f"  No tuning grid for {algo_name}, using fixed params: {params_to_run_with}")

            if algo_name == "MeanShift" and params_to_run_with.get("bandwidth") is None:
                n_s_bw = min(len(data_scaled), 500)
                ms_bw = 0.1 if n_s_bw < 2 else estimate_bandwidth(data_scaled, quantile=0.2, n_samples=n_s_bw, n_jobs=-1)
                params_to_run_with["bandwidth"] = max(ms_bw, 1e-3) if ms_bw is not None and np.isfinite(ms_bw) else 1.0
                print(f"  Auto-estimated MeanShift bandwidth for final run: {params_to_run_with['bandwidth']:.4f}")

            print(f"  Running {algo_name} with final params: {params_to_run_with}")
            param_text_list = [f"{k_p}: {v_p:.4f}" if (isinstance(v_p, float) or isinstance(v_p, np.float64)) and np.isfinite(v_p) else f"{k_p}: {v_p}" for k_p, v_p in params_to_run_with.items()]
            plot_text_params = f"{algo_name} (Params by {tuning_metric_name_display})\nPsi: {psi_estimator_name}\n" + "\n".join(param_text_list)

            clustering_instance = algo_class(**params_to_run_with)
            start_time_algo_run = time.time(); labels = clustering_instance.fit_predict(data_scaled); algo_duration = time.time() - start_time_algo_run
            n_clusters_found = len(np.unique(labels[(labels!=-1)&(labels!=-2)]))
            print(f"  {algo_name} completed in {algo_duration:.2f}s. Found {n_clusters_found} clusters.")

            data_to_plot_run = data_scaled[:,:2] if data_scaled.shape[1] > 2 else data_scaled
            dataset_name_for_plot = dataset_name
            if data_scaled.shape[1] > 2 and data_to_plot_run.shape[1] == 2:
                dataset_name_for_plot = f"{dataset_name} (2D Proj.)"

            if data_to_plot_run.shape[1] == 2:
                kde_for_plot_run = current_kde_psi_estimator
                if data_scaled.shape[1] > 2:
                     print(f"  Original data >2D. Creating 2D KDE for plot using: k={current_kde_psi_estimator.kernel}, bw_meth={kde_config['bandwidth_method']}")
                     kde_for_plot_run = create_kde_estimator(data_to_plot_run, current_kde_psi_estimator.kernel, kde_config['bandwidth_method'])

                plot_clusters_and_kde(data_to_plot_run, labels, kde_for_plot_run, dataset_name_for_plot,
                                      f"{algo_name}", psi_estimator_name_text=psi_estimator_name,
                                      clustering_params_text=plot_text_params)

            current_run_metrics = {"duration_clustering_s": algo_duration, "n_clusters_found": n_clusters_found,
                                   "final_params_used": str(params_to_run_with),
                                   "psi_estimator_config_name": psi_estimator_name,
                                   "tuned_by_primary_metric": tuning_metric_name_display,
                                   "primary_tuning_score": tuning_score_achieved if np.isfinite(tuning_score_achieved) else np.nan}

            can_calc_standard_metrics = n_clusters_found >= 2 and -2 not in np.unique(labels)
            if not can_calc_standard_metrics:
                current_run_metrics.update({"Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan, "DBCV": np.nan})
            else:
                current_run_metrics["Silhouette"] = silhouette_score(data_scaled, labels)
                current_run_metrics["Calinski-Harabasz"] = calinski_harabasz_score(data_scaled, labels)
                current_run_metrics["Davies-Bouldin"] = davies_bouldin_score(data_scaled, labels)
                try: current_run_metrics["DBCV"] = DBCV(data_scaled, labels, dist_function=euclidean)
                except Exception as e: current_run_metrics["DBCV"] = np.nan; print(f"  DBCV Error for {algo_name}: {e}")

            framework_scores_run = evaluate_clustering_framework(data_scaled, labels, current_kde_psi_estimator,
                                                                 mu_O_psi, sigma_O_psi, psi_O_values)
            current_run_metrics.update(framework_scores_run)

            overall_z_run = current_run_metrics.get('OverallClust_AvgDens_Z')
            sep_z_run = current_run_metrics.get('Sep_OverallMinPath_Z')
            if overall_z_run is not None and np.isfinite(overall_z_run):
                if n_clusters_found >= 2 and sep_z_run is not None and np.isfinite(sep_z_run): current_run_metrics['DDSI_psi'] = overall_z_run - sep_z_run
                elif n_clusters_found == 1: current_run_metrics['DDSI_psi'] = overall_z_run
                else: current_run_metrics['DDSI_psi'] = np.nan
            else: current_run_metrics['DDSI_psi'] = np.nan
            results_for_this_psi_estimator[algo_name] = current_run_metrics
        master_results_for_dataset_by_psi[psi_estimator_name] = results_for_this_psi_estimator

    for psi_name_key, results_for_this_psi in master_results_for_dataset_by_psi.items():
        if results_for_this_psi:
            results_df = pd.DataFrame(results_for_this_psi).T
            csv_filename = os.path.join(RESULTS_DIR, f"{dataset_name}_{psi_name_key}_evaluation_primary_tuned.csv")
            results_df.to_csv(csv_filename)
            print(f"\nResults for {dataset_name} (psi: {psi_name_key}, primary_tuned) saved to {csv_filename}")
            cols_to_print = ['tuned_by_primary_metric', 'primary_tuning_score',
                             'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'DBCV', 'DDSI_psi',
                             'OverallClust_AvgDens_Z', 'Outlier_AvgDens_Z', 'Sep_OverallMinPath_Z',
                             'final_params_used', 'n_clusters_found']
            existing_cols_to_print = [col for col in cols_to_print if col in results_df.columns]
            print(f"Summary for psi: {psi_name_key} (Dataset: {dataset_name})")
            print(results_df[existing_cols_to_print].round(4))
    total_duration = time.time() - start_time_total
    print(f"--- Experiment {dataset_name} completed in {total_duration:.2f}s ---")
    return master_results_for_dataset_by_psi

# --- Define Datasets and Fixed/Base Clustering Parameters ---
datasets_to_run = {}
fixed_clustering_params = {}
param_grids_for_tuning_generic = {
    "DBSCAN": {'eps': [0.1, 0.15, 0.2, 0.3, 0.5, 0.7], 'min_samples': [3, 5, 8, 10, 12]},
    "OPTICS": {'min_samples': [3, 5, 8, 12], 'xi': [0.01, 0.03, 0.05, 0.07, 0.1], 'min_cluster_size': [0.02, 0.05, 0.1]},
    "MeanShift": {'bandwidth': [None, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]},
}

# 1. Moons
n_samples_moons = 500
moons_data, _ = make_moons(n_samples=n_samples_moons, noise=0.12, random_state=42)
print(f"Generated Moons: {moons_data.shape[0]} objects, {moons_data.shape[1]} features.")
datasets_to_run["Moons"] = moons_data
fixed_clustering_params["Moons"] = {
    "DBSCAN": {"eps": 0.15, "min_samples": int(0.02*n_samples_moons)},
    "OPTICS": {"min_samples": int(0.02*n_samples_moons), "xi": 0.05, "min_cluster_size": 0.05},
    "MeanShift": {"bandwidth": None},
}

# 2. Aggregation
agg_data_full, _ = load_dataset("Aggregation.txt", DATA_DIR, delimiter='\t', skip_header=0, label_col=-1)
if agg_data_full is not None:
    datasets_to_run["Aggregation"] = agg_data_full
    fixed_clustering_params["Aggregation"] = {"DBSCAN": {"eps": 1.2, "min_samples": 6}, "OPTICS": {"min_samples": 6, "xi": 0.02, "min_cluster_size": 0.02}, "MeanShift": {"bandwidth": None}}
else: print("Aggregation dataset not loaded, skipping.")

# 3. R15
r15_data_full, _ = load_dataset("R15.txt", DATA_DIR, delimiter='\t', skip_header=0, label_col=-1)
if r15_data_full is not None:
    datasets_to_run["R15"] = r15_data_full
    fixed_clustering_params["R15"] = {"DBSCAN": {"eps": 0.65, "min_samples": 4}, "OPTICS": {"min_samples": 4, "xi": 0.015, "min_cluster_size": 0.01}, "MeanShift": {"bandwidth": None}}
else: print("R15 dataset not loaded, skipping.")

# 4. Seeds
seeds_data_full, _ = load_dataset('seeds_dataset.txt', DATA_DIR, delimiter='\s+', skip_header=0, label_col=-1)
if seeds_data_full is not None:
    datasets_to_run["Seeds"] = seeds_data_full
    fixed_clustering_params["Seeds"] = {"DBSCAN": {"eps": 0.8, "min_samples": 7}, "OPTICS": {"min_samples": 7, "xi": 0.04, "min_cluster_size": 0.05}, "MeanShift": {"bandwidth": None}}
else: print("Seeds dataset not loaded, skipping.")

# 5. Breast Cancer
bc_data_full, _ = load_dataset('brca.csv', DATA_DIR, delimiter=',', skip_header=1, label_col=-1) # Assuming header, label last
if bc_data_full is not None:
    datasets_to_run["BreastCancer"] = bc_data_full
    fixed_clustering_params["BreastCancer"] = {"DBSCAN": {"eps": 0.7, "min_samples": 10}, "OPTICS": {"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.05}, "MeanShift": {"bandwidth": None}}
else: print("Breast Cancer dataset not loaded, skipping.")

# 6. Glass
glass_data_full, _ = load_dataset('glass.csv', DATA_DIR, delimiter=',', skip_header=1, label_col=-1) # Assuming header, label last
if glass_data_full is not None:
    datasets_to_run["Glass"] = glass_data_full
    fixed_clustering_params["Glass"] = {"DBSCAN": {"eps": 0.5, "min_samples": 5}, "OPTICS": {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.1}, "MeanShift": {"bandwidth": None}}
else: print("Glass dataset not loaded, skipping.")

# 7. Wine Quality
wine_data_full, _ = load_dataset('Wine-qt.csv', DATA_DIR, delimiter=',', skip_header=1, label_col=-1) # Assuming header, label last
if wine_data_full is not None:
    datasets_to_run["Wine"] = wine_data_full
    fixed_clustering_params["Wine"] = {"DBSCAN": {"eps": 0.4, "min_samples": 8}, "OPTICS": {"min_samples": 8, "xi": 0.03, "min_cluster_size": 0.03}, "MeanShift": {"bandwidth": None}}
else: print("Wine dataset not loaded, skipping.")


def print_metric_table_demo(metrics_dict, title, float_format=".4f"):
    """
    Helper to print a dictionary as a nicely formatted table for the demo.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        title (str): Title for the table.
        float_format (str, optional): Format string for float values. Defaults to ".4f".
    """
    print(f"\n{title}")
    df_metrics = pd.DataFrame(metrics_dict.items(), columns=['Metric', 'Value'])
    df_metrics['Value'] = df_metrics['Value'].apply(lambda x: f"{x:{float_format}}" if isinstance(x, (float, np.float64)) and np.isfinite(x) else str(x))
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    master_results_collection = {}
    primary_tuning_metric_for_algos = {
        "name": "Silhouette",
        "func": silhouette_score,
        "higher_is_better": True
    }

    print(f"\n=== Starting Experiments for {len(datasets_to_run)} Datasets ===")
    for name, data_points in datasets_to_run.items():
        if data_points is None or len(data_points) == 0:
            print(f"Skipping dataset {name} due to loading issues or empty data.")
            master_results_collection[name] = {}
            continue

        current_fixed_params_for_this_dataset = fixed_clustering_params.get(name, {}).copy()

        if "OPTICS" in current_fixed_params_for_this_dataset:
            optics_p = current_fixed_params_for_this_dataset["OPTICS"]
            if "min_cluster_size" in optics_p and isinstance(optics_p["min_cluster_size"], float) and 0 < optics_p["min_cluster_size"] < 1:
                optics_p["min_cluster_size"] = max(2, int(optics_p["min_cluster_size"] * len(data_points)))

        current_param_grids_for_this_dataset = {k:v.copy() for k,v in param_grids_for_tuning_generic.items()}
        if "OPTICS" in current_param_grids_for_this_dataset and \
           "min_cluster_size" in current_param_grids_for_this_dataset["OPTICS"]:
            original_mcs_values = param_grids_for_tuning_generic["OPTICS"]["min_cluster_size"]
            current_param_grids_for_this_dataset["OPTICS"]["min_cluster_size"] = [
                max(2, int(frac_or_abs * len(data_points))) if isinstance(frac_or_abs, float) and 0 < frac_or_abs < 1 else max(2, int(frac_or_abs))
                for frac_or_abs in original_mcs_values
            ]

        master_results_collection[name] = run_experiment(
            name, data_points,
            current_fixed_params_for_this_dataset,
            current_param_grids_for_this_dataset,
            primary_tuning_metric_for_algos
        )

    print("\n\n=== All Experiments Complete ===")
    print("\n\n=== Summary: Best PSI Estimator Analysis (using avg DDSI_psi) ===")
    for dataset_name_key, results_by_psi in master_results_collection.items():
        if not results_by_psi: print(f"\n--- Dataset: {dataset_name_key} --- No results found or skipped."); continue
        best_avg_ddsi_score = -np.inf; best_psi_for_dataset = None
        print(f"\n--- Dataset: {dataset_name_key} ---")
        for psi_name, algo_results_map in results_by_psi.items():
            if not algo_results_map: continue
            ddsi_scores_for_this_psi = [metrics.get('DDSI_psi') for metrics in algo_results_map.values() if metrics.get('DDSI_psi') is not None and np.isfinite(metrics.get('DDSI_psi'))]
            if ddsi_scores_for_this_psi:
                current_avg_score = np.mean(ddsi_scores_for_this_psi)
                print(f"  Psi Estimator: {psi_name}, Avg DDSI_psi across algos: {current_avg_score:.4f}")
                if current_avg_score > best_avg_ddsi_score:
                    best_avg_ddsi_score, best_psi_for_dataset = current_avg_score, psi_name
            else: print(f"  Psi Estimator: {psi_name}, No valid DDSI_psi scores to average.")
        if best_psi_for_dataset: print(f"  >>> Best overall PSI (by avg DDSI_psi): {best_psi_for_dataset} (Score: {best_avg_ddsi_score:.4f})")
        else: print(f"  >>> Could not determine a best PSI for {dataset_name_key} based on DDSI_psi.")

    print("\n\n=== Preparing Data for DEMO Section ===")
    best_runs_for_demo = {}
    temp_scaler_demo = StandardScaler()

    for dataset_name_demo_key, data_raw_for_demo in datasets_to_run.items():
        if data_raw_for_demo is None or len(data_raw_for_demo) == 0:
            print(f"Skipping {dataset_name_demo_key} for demo preparation (no data).")
            continue

        print(f"\nPreparing demo data for: {dataset_name_demo_key}")
        data_scaled_for_demo = temp_scaler_demo.fit_transform(data_raw_for_demo)
        best_ddsi_for_this_dataset_demo = -np.inf
        details_of_best_run_for_demo = None

        if dataset_name_demo_key in master_results_collection and master_results_collection[dataset_name_demo_key]:
            for psi_estimator_name_demo, algo_results_map_demo in master_results_collection[dataset_name_demo_key].items():
                if not algo_results_map_demo: continue
                for algo_name_demo, metrics_dict_demo in algo_results_map_demo.items():
                    current_ddsi_demo = metrics_dict_demo.get('DDSI_psi', -np.inf)
                    if not np.isfinite(current_ddsi_demo): current_ddsi_demo = -np.inf

                    if current_ddsi_demo > best_ddsi_for_this_dataset_demo:
                        best_ddsi_for_this_dataset_demo = current_ddsi_demo
                        details_of_best_run_for_demo = {
                            'algo_name': algo_name_demo,
                            'psi_estimator_name': psi_estimator_name_demo,
                            'stored_metrics': metrics_dict_demo.copy()
                        }
        else:
            print(f"No results found in master_results_collection for {dataset_name_demo_key} to prepare demo.")
            continue

        if details_of_best_run_for_demo:
            print(f"  Found best run for {dataset_name_demo_key}: Algo {details_of_best_run_for_demo['algo_name']}, Psi {details_of_best_run_for_demo['psi_estimator_name']} with DDSI_psi: {best_ddsi_for_this_dataset_demo:.4f}")
            algo_name_best_demo = details_of_best_run_for_demo['algo_name']
            psi_name_best_demo = details_of_best_run_for_demo['psi_estimator_name']
            params_str_best_demo = details_of_best_run_for_demo['stored_metrics']['final_params_used']
            try:
                parsed_params_best_demo = ast.literal_eval(params_str_best_demo)
                if not isinstance(parsed_params_best_demo, dict): parsed_params_best_demo = {}
            except:
                print(f"    Warning: Could not parse params: {params_str_best_demo}. Using empty dict for {dataset_name_demo_key}.")
                parsed_params_best_demo = {}

            kde_config_best_demo = next((cfg for cfg in KDE_CONFIGURATIONS if cfg['name'] == psi_name_best_demo), None)
            if not kde_config_best_demo:
                print(f"    Error: KDE config {psi_name_best_demo} not found. Skipping demo for this run on {dataset_name_demo_key}.")
                continue

            print(f"    Recreating KDE for demo: {kde_config_best_demo['kernel']}, {kde_config_best_demo['bandwidth_method']}")
            recreated_kde_psi_estimator_demo = create_kde_estimator(data_scaled_for_demo, kde_config_best_demo['kernel'], kde_config_best_demo['bandwidth_method'])
            recreated_mu_O_psi_demo, recreated_sigma_O_psi_demo, recreated_psi_O_values_demo = calculate_dataset_density_stats(data_scaled_for_demo, recreated_kde_psi_estimator_demo)

            algo_class_best_demo = ALGO_DEFINITIONS[algo_name_best_demo]
            if algo_name_best_demo == "MeanShift" and parsed_params_best_demo.get("bandwidth") is None:
                n_s_bw_demo = min(len(data_scaled_for_demo), 500)
                ms_bw_demo = 0.1 if n_s_bw_demo < 2 else estimate_bandwidth(data_scaled_for_demo, quantile=0.2, n_samples=n_s_bw_demo, n_jobs=-1)
                parsed_params_best_demo["bandwidth"] = max(ms_bw_demo, 1e-3) if ms_bw_demo is not None and np.isfinite(ms_bw_demo) else 1.0
                print(f"    Re-estimated MeanShift bandwidth for demo run: {parsed_params_best_demo['bandwidth']:.4f}")

            print(f"    Re-running {algo_name_best_demo} for demo with params: {parsed_params_best_demo}")
            model_for_best_run_demo = algo_class_best_demo(**parsed_params_best_demo)
            labels_for_best_run_demo = model_for_best_run_demo.fit_predict(data_scaled_for_demo)
            n_clusters_found_for_demo = len(np.unique(labels_for_best_run_demo[labels_for_best_run_demo >= 0]))

            framework_scores_for_demo_run = evaluate_clustering_framework(
                data_scaled_for_demo, labels_for_best_run_demo,
                recreated_kde_psi_estimator_demo, recreated_mu_O_psi_demo,
                recreated_sigma_O_psi_demo, recreated_psi_O_values_demo
            )

            overall_metrics_for_demo_dict = framework_scores_for_demo_run.copy()
            overall_metrics_for_demo_dict['final_params_used'] = params_str_best_demo
            overall_metrics_for_demo_dict['n_clusters_found'] = n_clusters_found_for_demo

            ov_z_demo_recalc = overall_metrics_for_demo_dict.get('OverallClust_AvgDens_Z')
            s_z_demo_recalc = overall_metrics_for_demo_dict.get('Sep_OverallMinPath_Z') # Updated key
            ddsi_psi_demo_recalc = np.nan
            if ov_z_demo_recalc is not None and np.isfinite(ov_z_demo_recalc):
                if n_clusters_found_for_demo >= 2 and s_z_demo_recalc is not None and np.isfinite(s_z_demo_recalc):
                    ddsi_psi_demo_recalc = ov_z_demo_recalc - s_z_demo_recalc
                elif n_clusters_found_for_demo == 1:
                    ddsi_psi_demo_recalc = ov_z_demo_recalc
            overall_metrics_for_demo_dict['DDSI_psi'] = ddsi_psi_demo_recalc

            per_cluster_metrics_parsed_demo = {}
            unique_labels_from_best_run_demo = np.unique(labels_for_best_run_demo)
            cluster_ids_for_demo_run = [l for l in unique_labels_from_best_run_demo if l >= 0]

            for cid_demo in cluster_ids_for_demo_run:
                per_cluster_metrics_parsed_demo[cid_demo] = {
                    'Size': np.sum(labels_for_best_run_demo == cid_demo),
                    'AvgDens_psi': framework_scores_for_demo_run.get(f'C_{cid_demo}_AvgDens_psi'),
                    'AvgDens_Z': framework_scores_for_demo_run.get(f'C_{cid_demo}_AvgDens_Z'),
                    'DensVar_psi': framework_scores_for_demo_run.get(f'C_{cid_demo}_DensVar_psi')
                }

            tuned_by_text = details_of_best_run_for_demo['stored_metrics'].get('tuned_by_primary_metric', 'DDSI_psi criteria')
            data_to_plot_for_demo_run = data_scaled_for_demo[:, :2] if data_scaled_for_demo.shape[1] > 2 else data_scaled_for_demo
            dataset_name_for_demo_plot = dataset_name_demo_key
            if data_scaled_for_demo.shape[1] > 2 and data_to_plot_for_demo_run.shape[1] == 2:
                 dataset_name_for_demo_plot = f"{dataset_name_demo_key} (2D Proj.)"

            kde_estimator_for_demo_plot = recreated_kde_psi_estimator_demo
            if data_raw_for_demo.shape[1] > 2:
                print(f"    Original data for {dataset_name_demo_key} >2D. Creating 2D KDE for demo plot.")
                kde_estimator_for_demo_plot = create_kde_estimator(
                    data_to_plot_for_demo_run,
                    recreated_kde_psi_estimator_demo.kernel,
                    kde_config_best_demo['bandwidth_method']
                )

            best_runs_for_demo[dataset_name_demo_key] = {
                'data_raw': data_to_plot_for_demo_run,
                'labels': labels_for_best_run_demo,
                'kde_psi_estimator': kde_estimator_for_demo_plot,
                'dataset_name': dataset_name_for_demo_plot,
                'algo_name': algo_name_best_demo,
                'psi_estimator_name': psi_name_best_demo,
                'overall_metrics': overall_metrics_for_demo_dict,
                'plot_params_text': f"{algo_name_best_demo} (Selected by DDSI_psi from {tuned_by_text} tuning)\nPsi: {psi_name_best_demo}\n{params_str_best_demo}",
                'mu_O_psi': recreated_mu_O_psi_demo,
                'sigma_O_psi': recreated_sigma_O_psi_demo,
                'per_cluster_metrics': per_cluster_metrics_parsed_demo
            }
            print(f"    Successfully prepared demo data for {dataset_name_demo_key}.")
        else:
            print(f"  Could not find a best run (by DDSI_psi) for {dataset_name_demo_key} to include in demo.")

    # --- DEMO OF FRAMEWORK METRICS ---
    print("\n\n=== DEMO: Illustrating Evaluation Framework Metrics for Best Clusterings ===")
    if not best_runs_for_demo:
        print("\nNo runs available for the demo. This might be due to no valid DDSI_psi scores or issues in demo data preparation.")
    else:
        for dataset_name, run_details in best_runs_for_demo.items():
            print(f"\n--- Demo for Dataset: {run_details['dataset_name']} ---")
            print(f"Best Clustering (selected by DDSI_psi):")
            print(f"  Algorithm: {run_details['algo_name']}")
            print(f"  Psi Estimator: {run_details['psi_estimator_name']}")
            print(f"  Best Params Used: {run_details['overall_metrics']['final_params_used']}")
            print(f"  Number of Clusters Found: {run_details['overall_metrics']['n_clusters_found']}")
            ddsi_psi_val = run_details['overall_metrics']['DDSI_psi']
            ddsi_psi_str = f"{ddsi_psi_val:.4f}" if isinstance(ddsi_psi_val, (float, np.float64)) and np.isfinite(ddsi_psi_val) else str(ddsi_psi_val)
            print(f"  Overall DDSI_psi Score: {ddsi_psi_str}")

            print(f"\nPlot for Best Clustering in {run_details['dataset_name']}:")
            plot_clusters_and_kde(
                run_details['data_raw'],
                run_details['labels'],
                run_details['kde_psi_estimator'],
                run_details['dataset_name'],
                run_details['algo_name'],
                psi_estimator_name_text=run_details['psi_estimator_name'],
                clustering_params_text=run_details['plot_params_text']
            )
            plot_base_name = dataset_name 
            plot_placeholder = f"{plot_base_name}_{run_details['algo_name']}_{run_details['psi_estimator_name']}_plot.png"
            print(f"Plot placeholder: {plot_placeholder}")

            print(f"\nDataset-wide Density Statistics (for {run_details['psi_estimator_name']}):")
            print(f"  Mean Dataset-wide Density (mu_O_psi): {run_details['mu_O_psi']:.4f}")
            print(f"  Std Dev Dataset-wide Density (sigma_O_psi): {run_details['sigma_O_psi']:.4f}")

            print("\n--- All Evaluation Framework Metrics ---")
            overall_metrics_full_subset = {
                '1. Average density for union of all clusters CX (psi)': run_details['overall_metrics'].get('OverallClust_AvgDens_psi'),
                '2. Normalized (z-score) of average cluster density ($Z(\mu_{CX,\\psi})$)': run_details['overall_metrics'].get('OverallClust_AvgDens_Z'),
                '5a. Overall density variation for clustering X (std dev across clustered points, psi)': run_details['overall_metrics'].get('OverallClust_DensVar_psi'),
                '5b. Overall density variation for clustering X (weighted within, psi)': run_details['overall_metrics'].get('OverallClust_WeightedDensVar_psi'),
                '6. Average density for Out(X) (psi)': run_details['overall_metrics'].get('Outlier_AvgDens_psi'),
                '7. Normalized (z-score) for Average density for Out(X) ($Z(\mu_{Out(X),\\psi})$)': run_details['overall_metrics'].get('Outlier_AvgDens_Z'),
                '8. Minimum Density along a Straight Line Path (psi)': run_details['overall_metrics'].get('Sep_OverallMinPath_psi'), 
                '9. Overall Cluster Separation Metric ($Z(Sep_{min})$)': run_details['overall_metrics'].get('Sep_OverallMinPath_Z'),
                '10. DDSI_psi': run_details['overall_metrics'].get('DDSI_psi')
            }
            print_metric_table_demo(overall_metrics_full_subset, "Overall Metrics for Clustering:")

            print("\n--- Per-Cluster Density Results ---")
            per_cluster_table_data = []
            if run_details['per_cluster_metrics'] and isinstance(run_details['per_cluster_metrics'], dict):
                 try:
                     sorted_cluster_ids = sorted(
                         run_details['per_cluster_metrics'].keys(),
                         key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).replace('-', '').isdigit() else str(x)
                     )
                 except ValueError:
                     sorted_cluster_ids = sorted(run_details['per_cluster_metrics'].keys())

                 for cluster_id in sorted_cluster_ids:
                    metrics = run_details['per_cluster_metrics'][cluster_id]
                    per_cluster_table_data.append({
                        'Cluster ID': cluster_id,
                        'Size': metrics['Size'],
                        '1. Average density for cluster Ci (psi)': metrics['AvgDens_psi'],
                        '3. Normalized (z-score) of Avg Cluster Density ($Z(\mu_{Ci,\\psi})$)': metrics['AvgDens_Z'],
                        '4. Density standard deviation for cluster Ci (psi)': metrics['DensVar_psi']
                    })
            if per_cluster_table_data:
                df_per_cluster = pd.DataFrame(per_cluster_table_data)
                for col in ['1. Average density for cluster Ci (psi)', '3. Normalized (z-score) of Avg Cluster Density ($Z(\mu_{Ci,\\psi})$)', '4. Density standard deviation for cluster Ci (psi)']:
                    if col in df_per_cluster:
                         df_per_cluster[col] = df_per_cluster[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float64)) and np.isfinite(x) else str(x))
                print(df_per_cluster.to_string(index=False))
            else:
                print("No non-noise clusters found or per-cluster metrics unavailable.")
    print("\n\n=== DEMO COMPLETE ===")
