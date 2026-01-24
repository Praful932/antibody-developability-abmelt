#!/usr/bin/env python3
"""
Denormalize Tagg, Tm, and Tmon values from normalized holdout sets.

This script:
1. Loads reference file (tm_holdout_4.csv) to get merck_id and name mapping
2. Loads normalized values from tagg_holdout_normalized.csv, tm_holdout_normalized.csv, and tmon_holdout_normalized.csv
3. Filters to only include antibodies present in reference file
4. Denormalizes the normalized values using utils.py
5. Saves denormalized values to separate CSV files with _denormalized postfix

Note: The tmon file uses column 'tmonset' which represents T_mon_onset (tmon).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils import renormalize, DEFAULT_STATS

def denormalize_temperature_type(normalized_df, reference_df, temp_type, column_name, output_file):
    """
    Denormalize a specific temperature type.
    
    Args:
        normalized_df: DataFrame with normalized values
        reference_df: DataFrame with merck_id and name mapping
        temp_type: Temperature type ('tagg', 'tm', or 'tmon')
        column_name: Name of the column in normalized_df (e.g., 'tagg', 'tm', 'tmonset')
        output_file: Path to output CSV file
    
    Returns:
        DataFrame with denormalized values
    """
    print(f"\n{'='*80}")
    print(f"Processing {temp_type.upper()}")
    print(f"{'='*80}")
    
    # Check if column exists
    if column_name not in normalized_df.columns:
        print(f"ERROR: '{column_name}' column not found!")
        print(f"Available columns: {list(normalized_df.columns)}")
        return None
    
    # Filter normalized_df to only include antibodies present in reference file
    # The 'name' column in normalized_df contains Merck IDs
    print(f"\nFiltering to antibodies present in reference file...")
    filtered_df = normalized_df[normalized_df['name'].isin(reference_df['merck_id'])].copy()
    print(f"  Found {len(filtered_df)} matching antibodies")
    
    if len(filtered_df) == 0:
        print(f"ERROR: No matching antibodies found!")
        print(f"Reference antibodies (merck_id): {reference_df['merck_id'].tolist()}")
        print(f"Normalized antibodies (name): {normalized_df['name'].tolist()}")
        return None
    
    # Merge with reference to get merck_id and name
    merged_df = pd.merge(
        filtered_df[['name', column_name]],
        reference_df[['merck_id', 'name']],
        left_on='name',
        right_on='merck_id',
        how='inner'
    )
    
    # Denormalize the normalized values
    print(f"\nDenormalizing normalized {temp_type.upper()} values...")
    normalized_values = merged_df[column_name].values
    denormalized_values = renormalize(normalized_values, temp_type=temp_type)
    
    # Create output dataframe with merck_id, name, and denormalized value
    output_column = temp_type  # Use 'tmon' instead of 'tmonset' for output
    output_df = pd.DataFrame({
        'merck_id': merged_df['merck_id'],
        'name': merged_df['name_y'],
        output_column: denormalized_values
    })
    
    # Display results
    print(f"\nStatistics used:")
    print(f"  Mean: {DEFAULT_STATS[temp_type]['mean']:.2f}°C")
    print(f"  Std:  {DEFAULT_STATS[temp_type]['std']:.2f}°C")
    
    print(f"\n{'Merck ID':<15} {'Name':<20} {'Normalized':<15} {'Denormalized':<15}")
    print("-" * 65)
    
    for _, row in merged_df.iterrows():
        merck_id = row['merck_id']
        antibody_name = row['name_y']
        normalized_val = row[column_name]
        denormalized_val = output_df[output_df['merck_id'] == merck_id][output_column].values[0]
        print(f"{merck_id:<15} {antibody_name:<20} {normalized_val:<15.4f} {denormalized_val:<15.2f}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS")
    print(f"Mean Denormalized {temp_type.upper()}:     {denormalized_values.mean():.2f}°C")
    print(f"Std Denormalized {temp_type.upper()}:       {denormalized_values.std():.2f}°C")
    print(f"Min Denormalized {temp_type.upper()}:       {denormalized_values.min():.2f}°C")
    print(f"Max Denormalized {temp_type.upper()}:       {denormalized_values.max():.2f}°C")
    
    # Save results to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nDenormalized values saved to: {output_file}")
    
    return output_df

def compare_tm_values(actual_df, denormalized_df, normalized_df, data_dir):
    """
    Compare actual TM values with denormalized values.
    
    Args:
        actual_df: DataFrame with actual TM values (from tm_holdout_4.csv)
        denormalized_df: DataFrame with denormalized TM values
        normalized_df: DataFrame with normalized TM values
        data_dir: Path to data directory
    
    Returns:
        DataFrame with comparison results
    """
    print(f"\n{'='*80}")
    print("COMPARING ACTUAL vs DENORMALIZED TM VALUES")
    print(f"{'='*80}")
    
    # Merge actual, normalized, and denormalized values
    # First merge actual with normalized (on merck_id = name in normalized_df)
    temp_df = pd.merge(
        actual_df[['merck_id', 'name', 'tm']],
        normalized_df[['name', 'tm']],
        left_on='merck_id',
        right_on='name',
        how='inner',
        suffixes=('_actual', '_normalized')
    )
    
    # Rename columns from first merge
    temp_df = temp_df.rename(columns={
        'tm_actual': 'actual_tm',
        'tm_normalized': 'normalized_tm',
        'name_actual': 'antibody_name'
    })
    
    # Drop duplicate name column if it exists
    if 'name_normalized' in temp_df.columns:
        temp_df = temp_df.drop(columns=['name_normalized'])
    
    # Then merge with denormalized
    merged_df = pd.merge(
        temp_df,
        denormalized_df[['merck_id', 'tm']],
        on='merck_id',
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("ERROR: No matching antibodies found for comparison!")
        return None
    
    # Rename denormalized tm column
    merged_df = merged_df.rename(columns={
        'tm': 'denormalized_tm'
    })
    
    # Calculate errors
    merged_df['error'] = merged_df['denormalized_tm'] - merged_df['actual_tm']
    merged_df['abs_error'] = np.abs(merged_df['error'])
    merged_df['abs_error_percent'] = (merged_df['abs_error'] / merged_df['actual_tm']) * 100
    
    # Display results
    print(f"\nStatistics used for denormalization:")
    print(f"  Mean: {DEFAULT_STATS['tm']['mean']:.2f}°C")
    print(f"  Std:  {DEFAULT_STATS['tm']['std']:.2f}°C")
    
    print(f"\n{'Antibody':<20} {'Merck ID':<12} {'Actual TM':<12} {'Normalized':<12} {'Denormalized':<15} {'Error':<12} {'Abs Error':<12} {'Error %':<10}")
    print("-" * 110)
    
    for _, row in merged_df.iterrows():
        antibody_name = row['antibody_name']
        print(f"{antibody_name:<20} "
              f"{row['merck_id']:<12} "
              f"{row['actual_tm']:<12.2f} "
              f"{row['normalized_tm']:<12.4f} "
              f"{row['denormalized_tm']:<15.2f} "
              f"{row['error']:<12.2f} "
              f"{row['abs_error']:<12.2f} "
              f"{row['abs_error_percent']:<10.2f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nMean Absolute Error (MAE):     {merged_df['abs_error'].mean():.2f}°C")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt((merged_df['error']**2).mean()):.2f}°C")
    print(f"Mean Absolute Percent Error:    {merged_df['abs_error_percent'].mean():.2f}%")
    print(f"Max Absolute Error:             {merged_df['abs_error'].max():.2f}°C")
    print(f"Min Absolute Error:             {merged_df['abs_error'].min():.2f}°C")
    
    # Pearson correlation
    pearson_corr, pearson_pvalue = pearsonr(merged_df['actual_tm'], merged_df['denormalized_tm'])
    print(f"\nPearson Correlation (r):         {pearson_corr:.4f}")
    print(f"Pearson Correlation p-value:      {pearson_pvalue:.4f}")
    
    # Also show correlation using np.corrcoef for consistency
    correlation = np.corrcoef(merged_df['actual_tm'], merged_df['denormalized_tm'])[0, 1]
    print(f"Correlation (np.corrcoef):        {correlation:.4f}")
    
    # R-squared
    ss_res = np.sum((merged_df['actual_tm'] - merged_df['denormalized_tm'])**2)
    ss_tot = np.sum((merged_df['actual_tm'] - merged_df['actual_tm'].mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared (R²):                  {r_squared:.4f}")
    
    # Save results to CSV
    output_file = data_dir / "tm_comparison_results.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nComparison results saved to: {output_file}")
    
    return merged_df

def main():
    # File paths
    data_dir = Path(__file__).parent / "data" / "abmelt"
    reference_file = data_dir / "tm_holdout_4.csv"
    
    normalized_files = {
        'tagg': data_dir / "tagg_holdout_normalized.csv",
        'tm': data_dir / "tm_holdout_normalized.csv",
        'tmon': data_dir / "tmon_holdout_normalized.csv"
    }
    
    output_files = {
        'tagg': data_dir / "tagg_holdout_denormalized.csv",
        'tm': data_dir / "tm_holdout_denormalized.csv",
        'tmon': data_dir / "tmon_holdout_denormalized.csv"
    }
    
    column_names = {
        'tagg': 'tagg',
        'tm': 'tm',
        'tmon': 'tmonset'  # Note: column is named 'tmonset' not 'tmon'
    }
    
    # Load reference file to get merck_id and name mapping
    print("Loading reference file (tm_holdout_4.csv)...")
    reference_df = pd.read_csv(reference_file)
    print(f"  Found {len(reference_df)} antibodies in reference file")
    print(f"  Antibodies: {', '.join(reference_df['merck_id'].tolist())}")
    
    # Process each temperature type
    results = {}
    for temp_type in ['tagg', 'tm', 'tmon']:
        normalized_file = normalized_files[temp_type]
        output_file = output_files[temp_type]
        column_name = column_names[temp_type]
        
        # Load normalized values
        print(f"\n{'='*80}")
        print(f"Loading normalized {temp_type.upper()} values from {normalized_file.name}...")
        normalized_df = pd.read_csv(normalized_file)
        print(f"  Found {len(normalized_df)} antibodies with normalized {temp_type.upper()} values")
        
        # Denormalize
        result_df = denormalize_temperature_type(
            normalized_df, 
            reference_df, 
            temp_type, 
            column_name, 
            output_file
        )
        
        if result_df is not None:
            results[temp_type] = result_df
    
    # Compare actual vs denormalized TM values if TM was processed
    if 'tm' in results:
        # Load normalized TM values for comparison
        normalized_tm_df = pd.read_csv(normalized_files['tm'])
        compare_tm_values(reference_df, results['tm'], normalized_tm_df, data_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nSuccessfully denormalized {len(results)} temperature types:")
    for temp_type in results.keys():
        print(f"  - {temp_type.upper()}: {output_files[temp_type].name}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
