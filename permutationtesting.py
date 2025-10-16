#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Oct 19 20:03:01 2020
# @author: Ajit Johnson Nirmal
# Modified to work with CSV files and include structural elements

"""
Spatial Interaction Analysis Tool

This module quantifies spatial interactions between cell types and structural 
elements, assessing their co-localization beyond random chance, with support 
for both 2D and 3D datasets.
"""

# Import libraries
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from joblib import Parallel, delayed
import scipy
from functools import reduce


def spatial_interaction(csv_path,
                        x_coordinate='x',
                        y_coordinate='y',
                        z_coordinate=None,
                        cell_type='cell_type',
                        sample_id='sample',
                        method='radius', 
                        radius=30, 
                        knn=10,
                        permutation=1000,
                        subset=None,
                        pval_method='zscore',
                        structures=None,
                        structure_radius=None,
                        cell_types_for_normalization=None,
                        verbose=True,
                        output_path=None):
    """
    Quantifies spatial interactions between cell types and structural elements.
    
    Parameters:
        csv_path (str):  
            Path to CSV file containing cell data. Each row should represent a cell.

        x_coordinate (str, required):  
            Column name for the x-coordinates.

        y_coordinate (str, required):  
            Column name for the y-coordinates.

        z_coordinate (str, optional):  
            Column name for the z-coordinates, for 3D spatial data analysis.

        cell_type (str, required):  
            Column name indicating cell phenotype or any categorical cell classification.

        sample_id (str, required):  
            Column name for sample identifiers, useful for analysis within specific samples.

        method (str, optional):  
            Method to define neighborhoods: 'radius' for fixed distance, 'knn' for K nearest neighbors.

        radius (int, optional):  
            Radius for neighborhood definition (applies when method='radius').

        knn (int, optional):  
            Number of nearest neighbors to consider (applies when method='knn').

        permutation (int, optional):  
            Number of permutations for p-value calculation.

        subset (str, optional):  
            Specific sample identifier for targeted analysis. If None, analyzes all samples.

        pval_method (str, optional):  
            Method for p-value calculation: 
            - 'abs': absolute difference
            - 'zscore': z-score based significance
            - 'mine': empirical p-value calculation
        
        structures (dict, optional):
            Dictionary mapping structure names to their distance column names in the CSV.
            Example: {'Trabeculae': 'min_distance_Trabeculae', 
                     'Endothelium': 'min_distance_Endothelial',
                     'Fat': 'min_distance_Fat'}
            If None, no structural elements will be included.
        
        structure_radius (float, optional):
            Radius for determining proximity to structures. If None, uses the same radius as the 
            cell neighborhood radius. A small tolerance (0.000001) is added to account for floating point precision.
        
        cell_types_for_normalization (list, optional):
            List of cell types to use for normalization. If None, uses all cell types in the data.
            Example: ['Bcell', 'Tcell', 'Myeloid', 'Endothelial']
        
        verbose (bool):  
            If set to `True`, the function will print detailed messages about its progress.

        output_path (str, optional):
            Path to save the results as a CSV file. If None, results are not saved to disk.

    Returns:
        results_df (pandas.DataFrame):  
            DataFrame containing spatial interaction results with columns for phenotype pairs,
            interaction counts, and p-values for each sample.

    """
    
    
    def spatial_interaction_internal(data_subset,
                                    x_coordinate,
                                    y_coordinate,
                                    z_coordinate,
                                    cell_type,
                                    method, 
                                    radius, 
                                    knn,
                                    permutation, 
                                    sample_id,
                                    pval_method,
                                    structures,
                                    structure_radius,
                                    cell_types_for_normalization,
                                    current_sample):
        
        if verbose:
            print(f"Processing Sample: {current_sample}")
        
        # Set structure radius to match cell radius if not specified
        if structure_radius is None:
            structure_radius = radius
        
        # Create a dataFrame with the necessary information
        if z_coordinate is not None:
            if verbose:
                print("Including Z-axis")
            data = pd.DataFrame({
                'x': data_subset[x_coordinate], 
                'y': data_subset[y_coordinate], 
                'z': data_subset[z_coordinate], 
                'phenotype': data_subset[cell_type]
            })
        else:
            data = pd.DataFrame({
                'x': data_subset[x_coordinate], 
                'y': data_subset[y_coordinate], 
                'phenotype': data_subset[cell_type]
            })

        # Reset index to ensure proper mapping
        data = data.reset_index(drop=True)
        data_subset = data_subset.reset_index(drop=True)
        
        # Identify neighbourhoods based on the method used
        # a) KNN method
        if method == 'knn':
            if verbose:
                print(f"Identifying the {knn} nearest neighbours for every cell")
            if z_coordinate is not None:
                tree = BallTree(data[['x','y','z']], leaf_size=2)
                ind = tree.query(data[['x','y','z']], k=knn, return_distance=False)
            else:
                tree = BallTree(data[['x','y']], leaf_size=2)
                ind = tree.query(data[['x','y']], k=knn, return_distance=False)
            neighbours = pd.DataFrame(ind.tolist(), index=data.index)
            neighbours.drop(0, axis=1, inplace=True)  # Remove self neighbour
            
        # b) Local radius method
        elif method == 'radius':
            if verbose:
                print(f"Identifying neighbours within {radius} pixels of every cell")
            if z_coordinate is not None:
                kdt = BallTree(data[['x','y','z']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y','z']], r=radius, return_distance=False)
            else:
                kdt = BallTree(data[['x','y']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
                
            for i in range(0, len(ind)): 
                ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))  # remove self
            neighbours = pd.DataFrame(ind.tolist(), index=data.index)
        
        else:
            raise ValueError(f"Method must be 'knn' or 'radius', got '{method}'")
            
        # Map Phenotypes to Neighbours
        phenomap = dict(zip(list(range(len(data))), data['phenotype']))
        if verbose:
            print("Mapping phenotype to neighbors")
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')

        # Add structural elements as neighbors if specified
        if structures is not None:
            if verbose:
                print(f"Adding structural elements within {structure_radius} pixels as neighbors")
            
            for structure_name, distance_column in structures.items():
                # Check if distance column exists in the data
                if distance_column not in data_subset.columns:
                    if verbose:
                        print(f"Warning: {distance_column} not found in data, skipping {structure_name}")
                    continue
                
                # Add new column for this structure
                neighbours[len(neighbours.columns)] = np.nan
                
                # Assign structure name to cells within structure_radius
                close_cells_mask = data_subset[distance_column] < (structure_radius + 0.000001)
                close_cells_idx = data_subset[close_cells_mask].index
                neighbours.loc[close_cells_idx, len(neighbours.columns)-1] = structure_name
                
                if verbose:
                    print(f"  Added {close_cells_mask.sum()} {structure_name} neighbors")
    
        # Drop rows with no neighbours
        neighbours = neighbours.dropna(how='all')
        
        # Collapse all the neighbours into a single column
        n = pd.DataFrame(neighbours.stack(), columns=["neighbour_phenotype"])
        n.index = n.index.get_level_values(0)  # Drop the multi index
        
        # Merge with real phenotype
        n = n.merge(data['phenotype'], how='inner', left_index=True, right_index=True)
        
        # Permutation
        if verbose:
            print(f'Performing {permutation} permutations')
    
        def permutation_pval(data):
            data = data.assign(neighbour_phenotype=np.random.permutation(data['neighbour_phenotype']))
            data_freq = data.groupby(['phenotype','neighbour_phenotype'], observed=False).size().unstack()
            data_freq = data_freq.fillna(0).stack().values 
            return data_freq
        
        # Apply function in parallel
        final_scores = Parallel(n_jobs=-1)(delayed(permutation_pval)(data=n) for i in range(permutation)) 
        perm = pd.DataFrame(final_scores).T
        
        # Consolidate the permutation results
        if verbose:
            print('Consolidating the permutation results')
        
        # Calculate observed frequencies
        n_freq = n.groupby(['phenotype','neighbour_phenotype'], observed=False).size().unstack().fillna(0).stack() 
        
        # Permutation statistics
        mean = perm.mean(axis=1)
        std = perm.std(axis=1)
        
        # P-value calculation
        if pval_method == 'abs':
            # Absolute difference method
            p_values = abs(n_freq.values - mean) / (permutation + 1)
            p_values = p_values[~np.isnan(p_values)].values
            
        elif pval_method == 'zscore':
            # Z-score based method
            z_scores = (n_freq.values - mean) / std        
            z_scores[np.isnan(z_scores)] = 0
            p_values = scipy.stats.norm.sf(abs(z_scores)) * 2
            p_values = p_values[~np.isnan(p_values)]
            
        elif pval_method == "mine":
            # Empirical p-value calculation
            psum = np.zeros_like(perm.iloc[:,0])
            for c in range(len(perm.columns)):
                psum += ((n_freq.values - perm.iloc[:,c]) <= 0).astype(int)
            psum = (psum + 1) / (permutation + 1)
            p_values = psum.copy()
        
        else:
            raise ValueError(f"pval_method must be 'abs', 'zscore', or 'mine', got '{pval_method}'")
        
        # Compute Direction of interaction (interaction or avoidance)
        direction = ((n_freq.values - mean) / abs(n_freq.values - mean)).fillna(1)

        # Normalize based on total cell count
        k = n.groupby(['phenotype','neighbour_phenotype'], observed=False).size().unstack().fillna(0)
        
        # Add neighbour phenotypes that are not present to make k a square matrix
        columns_to_add = dict.fromkeys(np.setdiff1d(k.index, k.columns), 0)
        k = k.assign(**columns_to_add)

        # Get cell counts for normalization
        total_cell_count = data['phenotype'].value_counts()
        
        # If specific cell types for normalization are provided, use only those
        if cell_types_for_normalization is not None:
            available_types = [ct for ct in cell_types_for_normalization if ct in total_cell_count.index]
            if len(available_types) == 0:
                if verbose:
                    print("Warning: None of the specified cell types for normalization found. Using all cell types.")
                total_cell_count = total_cell_count.reindex(k.columns).fillna(1).values
            else:
                total_cell_count = total_cell_count[k.columns[k.columns.isin(available_types)]].values
        else:
            # Use all cell types present
            total_cell_count = total_cell_count.reindex(k.columns).fillna(1).values

        # Normalize
        k_max = k.div(total_cell_count, axis=0)
        k_max = k_max.div(k_max.max(axis=1), axis=0).stack()

        # Create DataFrame with the neighbour frequency and P values
        try:
            count = (k_max.values * direction).values  # adding directionality to interaction
        except Exception as e:
            if verbose:
                print(f"Error in normalization: {e}")
            # Return empty result structure
            result = pd.DataFrame({
                'phenotype': [],
                'neighbour_phenotype': [],
                current_sample: [],
                f'pvalue_{current_sample}': []
            })
            return result

        result = pd.DataFrame({
            'count': count,
            'p_val': np.array(p_values)
        }, index=k_max.index)

        result.columns = [current_sample, f'pvalue_{current_sample}']
        result = result.reset_index()
        
        # Display results
        if verbose:
            print("\nResults summary:")
            print(result.head(10))
            if structures is not None:
                struct_results = result[result['neighbour_phenotype'].isin(structures.keys())]
                if len(struct_results) > 0:
                    print("\nStructural interactions found:")
                    print(struct_results)
        
        return result
    
    
    # Load the CSV file
    if verbose:
        print(f"Loading data from {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = [x_coordinate, y_coordinate, cell_type, sample_id]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if z_coordinate is not None and z_coordinate not in data.columns:
        raise ValueError(f"Z-coordinate column '{z_coordinate}' not found in data")
    
    if verbose:
        print(f"Data loaded: {len(data)} cells across {data[sample_id].nunique()} samples")
        print(f"Cell types found: {data[cell_type].unique()}")
    
    # Subset a particular sample if requested, otherwise process all samples
    if subset is not None:
        if subset not in data[sample_id].unique():
            raise ValueError(f"Subset '{subset}' not found in {sample_id} column")
        sample_list = [subset]
        data_list = [data[data[sample_id] == subset]]
    else:
        sample_list = data[sample_id].unique()
        data_list = [data[data[sample_id] == i] for i in sample_list]
    
    if verbose:
        print(f"Processing {len(sample_list)} sample(s)")
    
    # Apply function to all samples and create a master dataframe
    all_data = []
    for i, (sample_data, sample_name) in enumerate(zip(data_list, sample_list)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{len(sample_list)}")
            print(f"{'='*60}")
        
        result = spatial_interaction_internal(
            data_subset=sample_data,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
            z_coordinate=z_coordinate,
            cell_type=cell_type,
            method=method,
            radius=radius,
            knn=knn,
            permutation=permutation,
            sample_id=sample_id,
            pval_method=pval_method,
            structures=structures,
            structure_radius=structure_radius,
            cell_types_for_normalization=cell_types_for_normalization,
            current_sample=sample_name
        )
        all_data.append(result)
    
    # Merge all the results into a single dataframe    
    if len(all_data) > 1:
        df_merged = reduce(lambda left, right: pd.merge(
            left, right, 
            on=['phenotype', 'neighbour_phenotype'], 
            how='outer'
        ), all_data)
    else:
        df_merged = all_data[0]
    
    if verbose:
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")
        print(f"Results shape: {df_merged.shape}")
    
    # Save to file if output path is provided
    if output_path is not None:
        df_merged.to_csv(output_path, index=False)
        if verbose:
            print(f"Results saved to {output_path}")
    
    # Return results
    return df_merged


def analyze_spatial_interactions(csv_path, 
                                 structures_dict=None,
                                 radius=30,
                                 **kwargs):
    """
    Convenience wrapper for spatial_interaction with sensible defaults.
    
    Parameters:
        csv_path (str): Path to CSV file
        structures_dict (dict): Dictionary of structure names to distance columns
        radius (int): Neighborhood radius
        **kwargs: Additional arguments passed to spatial_interaction
    
    Returns:
        pandas.DataFrame: Results of spatial interaction analysis
    """
    return spatial_interaction(
        csv_path=csv_path,
        structures=structures_dict,
        radius=radius,
        method='radius',
        permutation=1000,
        pval_method='zscore',
        **kwargs
    )

# Define your structures
structures = {
    'Trabeculae': 'min_distance_Trabeculae',
    'Endothelium': 'min_distance_Endothelial',
    'Fat': 'min_distance_Fat'
}

# Call the function
results = analyze_spatial_interactions(
    csv_path='cells.csv',
    structures_dict=structures,
    radius=200,
    x_coordinate='center_x',
    y_coordinate='center_y',
    cell_type='cell_type',
    sample_id='sample_name'
)