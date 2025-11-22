#!/usr/bin/env python3

"""
Descriptor computation module for AbMelt inference pipeline.
Handles extraction of MD descriptors from trajectories and aggregation into ML-ready format.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import glob

# Add the original AbMelt src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "AbMelt" / "src"))

try:
    import gromacs
    from order_param import order_s2, avg_s2_blocks, get_lambda, order_lambda
    from res_sasa import core_surface, get_core_surface, get_slope
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

logger = logging.getLogger(__name__)


def compute_descriptors(simulation_result: Dict, config: Dict) -> Dict:
    """
    Main entry point for descriptor computation.
    
    Args:
        simulation_result: Dictionary containing trajectory files and work directory
        config: Configuration dictionary
        
    Returns:
        Dictionary containing descriptors DataFrame and metadata
    """
    logger.info("Starting descriptor computation...")
    
    work_dir = Path(simulation_result["work_dir"])
    trajectory_files = simulation_result["trajectory_files"]
    antibody_name = work_dir.name
    
    # Get descriptor computation parameters
    desc_config = config["descriptors"]
    eq_time = desc_config["equilibration_time"]
    block_length = desc_config["block_length"]
    core_surface_k = desc_config["core_surface_k"]
    compute_lambda = desc_config["compute_lambda"]
    
    # Extract temperatures from trajectory files
    temps = [str(temp) for temp in trajectory_files.keys()]
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to work directory
        os.chdir(work_dir)
        logger.info(f"Changed to work directory: {work_dir}")
        
        # Step 1: Compute GROMACS-based descriptors
        logger.info("Step 1: Computing GROMACS descriptors...")
        xvg_files = _compute_gromacs_descriptors(work_dir, temps, eq_time)
        logger.info(f"Generated {len(xvg_files)} GROMACS descriptor files")
        
        # Step 2: Compute order parameters
        logger.info("Step 2: Computing order parameters...")
        master_s2_dict = _compute_order_parameters(work_dir, temps, eq_time, block_length, antibody_name)
        
        # Step 3: Compute core/surface SASA
        logger.info("Step 3: Computing core/surface SASA...")
        sasa_dict = _compute_core_surface_sasa(work_dir, temps, eq_time, core_surface_k)
        
        # Step 4: Compute multi-temperature features (lambda)
        if len(temps) >= 2 and compute_lambda:
            logger.info("Step 4: Computing multi-temperature lambda...")
            lambda_dict, r_dict = _compute_lambda_features(master_s2_dict, temps, block_length, eq_time, antibody_name)
        else:
            logger.warning(f"Skipping lambda computation: need >=2 temperatures, got {len(temps)}")
            lambda_dict = None
            r_dict = None
        
        # Step 5: Aggregate all descriptors into DataFrame
        logger.info("Step 5: Aggregating descriptors to DataFrame...")
        descriptors_df = _aggregate_descriptors_to_dataframe(
            work_dir, temps, antibody_name, eq_time, master_s2_dict, 
            lambda_dict, r_dict, sasa_dict, core_surface_k
        )
        
        logger.info(f"Descriptor computation completed. DataFrame shape: {descriptors_df.shape}")
        logger.info(f"Features: {list(descriptors_df.columns)}")
        
        # Save descriptors to file for future use
        try:
            descriptors_csv = work_dir / "descriptors.csv"
            descriptors_pkl = work_dir / "descriptors.pkl"
            
            descriptors_df.to_csv(descriptors_csv, index=False)
            logger.info(f"Saved descriptors to {descriptors_csv}")
            
            import pickle
            with open(descriptors_pkl, 'wb') as f:
                pickle.dump(descriptors_df, f)
            logger.info(f"Saved descriptors to {descriptors_pkl}")
        except Exception as e:
            logger.warning(f"Failed to save descriptors to file: {e}")
            logger.warning("Continuing without saving descriptors")
        
        result = {
            "status": "success",
            "descriptors_df": descriptors_df,
            "xvg_files": xvg_files,
            "work_dir": str(work_dir),
            "message": "Descriptor computation completed successfully"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Descriptor computation failed: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def _compute_gromacs_descriptors(work_dir: Path, temps: List[str], eq_time: int) -> List[str]:
    """
    Compute GROMACS-based descriptors from trajectories.
    
    Args:
        work_dir: Working directory containing trajectories
        temps: List of temperature strings
        eq_time: Equilibration time in ns
        
    Returns:
        List of generated .xvg file paths
    """
    xvg_files = []
    
    for temp in temps:
        logger.info(f"Computing GROMACS descriptors for {temp}K...")
        
        # Check if required files exist
        final_xtc = f'md_final_{temp}.xtc'
        final_gro = f'md_final_{temp}.gro'
        tpr_file = f'md_{temp}.tpr'
        index_file = 'index.ndx'
        
        if not os.path.exists(final_xtc):
            raise ValueError(f'Trajectory file not found: {final_xtc}')
        if not os.path.exists(final_gro):
            raise ValueError(f'Structure file not found: {final_gro}')
        if not os.path.exists(tpr_file):
            raise ValueError(f'TPR file not found: {tpr_file}')
        if not os.path.exists(index_file):
            logger.warning(f'Index file not found: {index_file}. CDR-specific features may fail.')
            # Try to create index file if it doesn't exist
            # This should have been created during MD simulation, but handle gracefully
            logger.warning('Attempting to create index file...')
            try:
                from preprocess import canonical_index
                annotation = canonical_index(pdb='processed.pdb')
                gromacs.make_ndx(f='processed.gro', o='index.ndx', input=annotation)
                logger.info('Index file created successfully')
            except Exception as e:
                logger.error(f'Failed to create index file: {e}')
                logger.error('CDR-specific features will be skipped')
        
        try:
            # Global features
            logger.debug(f"  Computing global SASA...")
            gromacs.sasa(f=final_xtc, s=final_gro, o=f'sasa_{temp}.xvg', input=['1'])
            xvg_files.append(f'sasa_{temp}.xvg')
            
            logger.debug(f"  Computing hydrogen bonds...")
            gromacs.hbond(f=final_xtc, s=tpr_file, num=f'bonds_{temp}.xvg', input=['1', '1'])
            xvg_files.append(f'bonds_{temp}.xvg')
            
            logger.debug(f"  Computing RMSD...")
            gromacs.rms(f=final_xtc, s=final_gro, o=f'rmsd_{temp}.xvg', input=['3', '3'])
            xvg_files.append(f'rmsd_{temp}.xvg')
            
            logger.debug(f"  Computing gyration radius...")
            gromacs.gyrate(f=final_xtc, s=final_gro, o=f'gyr_{temp}.xvg', n=index_file, input=['1'])
            xvg_files.append(f'gyr_{temp}.xvg')
            
            # CDR-specific features
            cdr_regions = {
                'cdrl1': '12',
                'cdrl2': '13',
                'cdrl3': '14',
                'cdrh1': '15',
                'cdrh2': '16',
                'cdrh3': '17',
                'cdrs': '18'
            }
            
            # SASA for each CDR
            logger.debug(f"  Computing CDR SASA...")
            for cdr_name, index_group in cdr_regions.items():
                gromacs.sasa(f=final_xtc, s=final_gro, o=f'sasa_{cdr_name}_{temp}.xvg', 
                           n=index_file, input=[index_group])
                xvg_files.append(f'sasa_{cdr_name}_{temp}.xvg')
            
            # H-bonds between light and heavy chains
            logger.debug(f"  Computing light-heavy bonds...")
            gromacs.hbond(f=final_xtc, s=tpr_file, num=f'bonds_lh_{temp}.xvg', 
                         n=index_file, input=['10', '11'])
            xvg_files.append(f'bonds_lh_{temp}.xvg')
            
            # RMSF for each CDR
            logger.debug(f"  Computing CDR RMSF...")
            eq_time_ps = str(eq_time * 1000)
            for cdr_name, index_group in cdr_regions.items():
                gromacs.rmsf(f=final_xtc, s=final_gro, o=f'rmsf_{cdr_name}_{temp}.xvg',
                           n=index_file, b=eq_time_ps, input=[index_group, index_group])
                xvg_files.append(f'rmsf_{cdr_name}_{temp}.xvg')
            
            # Gyration radius for each CDR
            logger.debug(f"  Computing CDR gyration...")
            for cdr_name, index_group in cdr_regions.items():
                gromacs.gyrate(f=final_xtc, s=final_gro, o=f'gyr_{cdr_name}_{temp}.xvg',
                             n=index_file, input=[index_group])
                xvg_files.append(f'gyr_{cdr_name}_{temp}.xvg')
            
            # Conformational entropy (S_conf)
            logger.debug(f"  Computing conformational entropy...")
            try:
                gromacs.trjconv(f=final_xtc, s=tpr_file, dt='0', fit='rot+trans', 
                             n=index_file, o=f'md_final_covar_{temp}.xtc', input=['1','1'])
                gromacs.covar(f=f'md_final_covar_{temp}.xtc', s=tpr_file, n=index_file,
                            o=f'covar_{temp}.xvg', av=f'avg_covar{temp}.pdb',
                            ascii=f'covar_matrix_{temp}.dat', v=f'covar_{temp}.trr',
                            input=['4', '4'])
                # Note: anaeig output goes to log file via shell redirection
                # The original implementation uses shell redirection in input parameter
                gromacs.anaeig(f=f'md_final_covar_{temp}.xtc', v=f'covar_{temp}.trr',
                             entropy=True, temp=temp, s=tpr_file, nevskip='6',
                             n=index_file, b=eq_time_ps, input=[f'> sconf_{temp}.log'])
            except Exception as e:
                logger.warning(f"Conformational entropy computation failed for {temp}K: {e}")
            
            # Electrostatic potential for each CDR
            logger.debug(f"  Computing CDR electrostatic potential...")
            for cdr_name, index_group in cdr_regions.items():
                try:
                    gromacs.potential(f=final_xtc, s=tpr_file, spherical=True, sl='10',
                                   o=f'potential_{cdr_name}_{temp}.xvg',
                                   oc=f'charge_{cdr_name}_{temp}.xvg',
                                   of=f'field_{cdr_name}_{temp}.xvg',
                                   n=index_file, input=[index_group])
                    xvg_files.append(f'potential_{cdr_name}_{temp}.xvg')
                except Exception as e:
                    logger.warning(f"Potential computation failed for {cdr_name} at {temp}K: {e}")
            
            # Dipole moment
            logger.debug(f"  Computing dipole moment...")
            gromacs.dipoles(f=final_xtc, s=tpr_file, o=f'dipole_{temp}.xvg',
                          n=index_file, input=['1'])
            xvg_files.append(f'dipole_{temp}.xvg')
            
        except Exception as e:
            logger.error(f"Failed to compute GROMACS descriptors for {temp}K: {e}")
            raise
    
    return xvg_files


def _compute_order_parameters(work_dir: Path, temps: List[str], eq_time: int, 
                             block_length: int, antibody_name: str) -> Dict[int, Dict]:
    """
    Compute N-H bond order parameters (S²) for each temperature.
    
    Args:
        work_dir: Working directory
        temps: List of temperature strings
        eq_time: Equilibration time in ns
        block_length: Block length for order parameter calculation in ns
        antibody_name: Name of antibody
        
    Returns:
        Dictionary mapping temperature (int) to S² values per residue
    """
    master_s2_dict = {int(temp): {} for temp in temps}
    
    for temp in temps:
        logger.info(f"Computing order parameters for {temp}K...")
        
        final_xtc = f'md_final_{temp}.xtc'
        final_gro = f'md_final_{temp}.gro'
        
        if not os.path.exists(final_xtc) or not os.path.exists(final_gro):
            logger.warning(f"Trajectory files not found for {temp}K, skipping order parameters")
            continue
        
        try:
            s2_blocks_dict = order_s2(mab=antibody_name, temp=temp, 
                                    block_length=block_length, start=eq_time)
            master_s2_dict[int(temp)] = avg_s2_blocks(s2_blocks_dict)
            logger.info(f"Order parameters computed for {temp}K")
        except Exception as e:
            logger.warning(f"Order parameter computation failed for {temp}K: {e}")
            logger.warning("This is common with short trajectories. Continuing...")
    
    return master_s2_dict


def _compute_core_surface_sasa(work_dir: Path, temps: List[str], eq_time: int, k: int) -> Dict:
    """
    Compute core/surface SASA using mdtraj.
    
    Args:
        work_dir: Working directory
        temps: List of temperature strings
        eq_time: Equilibration time in ns
        k: Number of residues for core/surface classification
        
    Returns:
        Dictionary with SASA statistics per temperature
    """
    sasa_dict = {}
    
    for temp in temps:
        logger.info(f"Computing core/surface SASA for {temp}K...")
        
        final_xtc = f'md_final_{temp}.xtc'
        final_gro = f'md_final_{temp}.gro'
        
        if not os.path.exists(final_xtc) or not os.path.exists(final_gro):
            logger.warning(f"Trajectory files not found for {temp}K, skipping SASA")
            continue
        
        try:
            # Compute residue-level SASA
            core_surface(temp)
            
            # Aggregate statistics
            sasa_dict[temp] = {}
            sasa_dict = get_core_surface(sasa_dict, temp, k=k, start=eq_time)
            logger.info(f"Core/surface SASA computed for {temp}K")
        except Exception as e:
            logger.warning(f"Core/surface SASA computation failed for {temp}K: {e}")
    
    return sasa_dict


def _compute_lambda_features(master_s2_dict: Dict[int, Dict], temps: List[str],
                            block_length: int, eq_time: int, antibody_name: str) -> Tuple[Dict, Dict]:
    """
    Compute multi-temperature lambda (order parameter slope).
    
    Args:
        master_s2_dict: Dictionary of S² values per temperature
        temps: List of temperature strings
        block_length: Block length used for order parameter calculation
        eq_time: Equilibration time
        antibody_name: Name of antibody
        
    Returns:
        Tuple of (lambda_dict, r_dict) - lambda values and correlation coefficients per residue
    """
    # Convert temps to ints for consistency
    temp_ints = [int(t) for t in temps]
    
    # Filter out temperatures that don't have S² data
    available_temps = [t for t in temp_ints if t in master_s2_dict and len(master_s2_dict[t]) > 0]
    
    if len(available_temps) < 2:
        logger.warning(f"Need at least 2 temperatures with S² data for lambda, got {len(available_temps)}")
        return None, None
    
    try:
        # Use order_lambda function from order_param module (saves CSV)
        # Note: start parameter expects picoseconds
        order_lambda(master_dict=master_s2_dict, mab=antibody_name, 
                    temps=available_temps, block_length=str(block_length), 
                    start=str(eq_time * 1000))
        
        # Compute lambda and r for each residue directly
        lambda_dict, r_dict = get_lambda(master_s2_dict, temps=available_temps)
        
        logger.info(f"Lambda computed for {len(lambda_dict)} residues")
        return lambda_dict, r_dict
        
    except Exception as e:
        logger.warning(f"Lambda computation failed: {e}")
        return None, None


def _aggregate_descriptors_to_dataframe(work_dir: Path, temps: List[str], 
                                       antibody_name: str, eq_time: int,
                                       master_s2_dict: Dict, lambda_dict: Optional[Dict],
                                       r_dict: Optional[Dict], sasa_dict: Dict,
                                       core_surface_k: int) -> pd.DataFrame:
    """
    Aggregate all computed descriptors into a single-row DataFrame.
    
    Args:
        work_dir: Working directory
        temps: List of temperature strings
        antibody_name: Name of antibody
        eq_time: Equilibration time in ns
        master_s2_dict: Order parameter dictionary
        lambda_dict: Lambda values dictionary (optional)
        r_dict: Correlation coefficients dictionary (optional)
        sasa_dict: Core/surface SASA dictionary
        core_surface_k: Number of residues for core/surface classification
        
    Returns:
        Single-row DataFrame with all descriptors
    """
    descriptor_dict = {}
    
    # Parse all .xvg files
    xvg_files = glob.glob('*.xvg')
    
    for xvg_file in xvg_files:
        try:
            metric_name = Path(xvg_file).stem
            
            # Extract temperature from filename
            temp = None
            for t in temps:
                if t in metric_name:
                    temp = t
                    break
            
            if temp is None:
                continue
            
            # Parse the xvg file
            data = _parse_xvg_file(xvg_file)
            
            if data is None or len(data) == 0:
                continue
            
            # Compute equilibrated statistics
            eq_time_ps = eq_time * 1000  # Convert to ps
            eq_start_idx = int(eq_time_ps / 10)  # Assuming 10 ps per frame (adjust if needed)
            
            if len(data) > eq_start_idx:
                equilibrated_data = data[eq_start_idx:]
                
                # Handle different data shapes
                if equilibrated_data.ndim == 1:
                    # Single column data
                    mu = np.mean(equilibrated_data)
                    std = np.std(equilibrated_data)
                    
                    # Create feature names based on metric type
                    # Match exact naming conventions from training data
                    if 'bonds' in metric_name:
                        if 'lh' in metric_name:
                            descriptor_dict[f'bonds_lh_mu_{temp}'] = mu
                            descriptor_dict[f'bonds_lh_std_{temp}'] = std
                        else:
                            # bonds file has hbonds and contacts - handled in 2D case
                            descriptor_dict[f'bonds_hbonds_mu_{temp}'] = mu
                            descriptor_dict[f'bonds_hbonds_std_{temp}'] = std
                    elif 'sasa' in metric_name:
                        region = metric_name.replace('sasa_', '').replace(f'_{temp}', '')
                        descriptor_dict[f'sasa_{region}_mu_{temp}'] = mu
                        descriptor_dict[f'sasa_{region}_std_{temp}'] = std
                    elif 'rmsd' in metric_name:
                        descriptor_dict[f'rmsd_mu_{temp}'] = mu
                        descriptor_dict[f'rmsd_std_{temp}'] = std
                    elif 'rmsf' in metric_name:
                        region = metric_name.replace('rmsf_', '').replace(f'_{temp}', '')
                        # Some models use mu, some use std - include both
                        descriptor_dict[f'rmsf_{region}_mu_{temp}'] = mu
                        descriptor_dict[f'rmsf_{region}_std_{temp}'] = std
                    elif 'gyr' in metric_name:
                        region = metric_name.replace('gyr_', '').replace(f'_{temp}', '')
                        # Training data shows: gyr_cdrs_Rg_std_350, gyr_cdrs_Rg_std_400
                        descriptor_dict[f'gyr_{region}_Rg_mu_{temp}'] = mu
                        descriptor_dict[f'gyr_{region}_Rg_std_{temp}'] = std
                    elif 'potential' in metric_name:
                        region = metric_name.replace('potential_', '').replace(f'_{temp}', '')
                        # Use mean value (potential at specific radius)
                        descriptor_dict[f'potential_{region}_mu_{temp}'] = mu
                    elif 'dipole' in metric_name:
                        descriptor_dict[f'dipole_mu_{temp}'] = mu
                        descriptor_dict[f'dipole_std_{temp}'] = std
                
                elif equilibrated_data.ndim == 2:
                    # Multi-column data (e.g., gyration with Rg, Rx, Ry, Rz)
                    if equilibrated_data.shape[1] >= 4:
                        # Gyration radius components
                        if 'gyr' in metric_name:
                            region = metric_name.replace('gyr_', '').replace(f'_{temp}', '')
                            r_values = equilibrated_data[:, 0]  # Rg
                            x_values = equilibrated_data[:, 1]  # Rx
                            y_values = equilibrated_data[:, 2]  # Ry
                            z_values = equilibrated_data[:, 3]  # Rz
                            
                            # Match training data format: gyr_cdrs_Rg_std_350
                            descriptor_dict[f'gyr_{region}_Rg_mu_{temp}'] = np.mean(r_values)
                            descriptor_dict[f'gyr_{region}_Rg_std_{temp}'] = np.std(r_values)
                            descriptor_dict[f'gyr_{region}_Rx_mu_{temp}'] = np.mean(x_values)
                            descriptor_dict[f'gyr_{region}_Rx_std_{temp}'] = np.std(x_values)
                            descriptor_dict[f'gyr_{region}_Ry_mu_{temp}'] = np.mean(y_values)
                            descriptor_dict[f'gyr_{region}_Ry_std_{temp}'] = np.std(y_values)
                            descriptor_dict[f'gyr_{region}_Rz_mu_{temp}'] = np.mean(z_values)
                            descriptor_dict[f'gyr_{region}_Rz_std_{temp}'] = np.std(z_values)
                    
                    elif equilibrated_data.shape[1] == 2:
                        # Two-column data (e.g., bonds with hbonds and contacts)
                        if 'bonds' in metric_name:
                            hbonds = equilibrated_data[:, 0]
                            contacts = equilibrated_data[:, 1]
                            
                            # Match training data format: bonds_contacts_std_350
                            descriptor_dict[f'bonds_hbonds_mu_{temp}'] = np.mean(hbonds)
                            descriptor_dict[f'bonds_hbonds_std_{temp}'] = np.std(hbonds)
                            descriptor_dict[f'bonds_contacts_mu_{temp}'] = np.mean(contacts)
                            descriptor_dict[f'bonds_contacts_std_{temp}'] = np.std(contacts)
        
        except Exception as e:
            logger.warning(f"Failed to parse {xvg_file}: {e}")
            continue
    
    # Add order parameter features
    for temp_int, s2_values in master_s2_dict.items():
        if s2_values and len(s2_values) > 0:
            temp_str = str(temp_int)
            s2_mean = np.mean(list(s2_values.values()))
            s2_std = np.std(list(s2_values.values()))
            descriptor_dict[f'order_s2_{temp_str}_mu'] = s2_mean
            descriptor_dict[f'order_s2_{temp_str}_std'] = s2_std
    
    # Add lambda features
    # Match training data formats:
    # - tagg: all-temp_lamda_b=25_eq=20
    # - tmon: r-lamda_b=2.5_eq=20
    if lambda_dict and r_dict:
        lambda_mean = np.mean(list(lambda_dict.values()))
        r_mean = np.mean(list(r_dict.values()))
        # Use block_length as shown in training data (may need adjustment)
        descriptor_dict[f'all-temp_lamda_b={block_length}_eq={eq_time}'] = lambda_mean
        descriptor_dict[f'r-lamda_b={block_length}_eq={eq_time}'] = lambda_mean  # For tmon model
        descriptor_dict[f'all-temp_lamda_r_b={block_length}_eq={eq_time}'] = r_mean
    
    # Add core/surface SASA features
    if sasa_dict:
        # Per-temperature SASA features
        for temp, sasa_data in sasa_dict.items():
            if isinstance(sasa_data, dict):
                for key, value in sasa_data.items():
                    descriptor_dict[f'sasa_{key}_{temp}'] = value
        
        # Cross-temperature SASA slopes
        if len(temps) >= 2:
            temp_ints = sorted([int(t) for t in temps])
            sasa_slopes = {}
            
            for key in ['total_mean', 'core_mean', 'surface_mean', 'total_std', 'core_std', 'surface_std']:
                data_points = [(int(t), sasa_dict[t][key]) for t in temps if t in sasa_dict and key in sasa_dict[t]]
                if len(data_points) >= 2:
                    slope = get_slope(data_points)
                    sasa_slopes[key] = slope
            
            for key, slope in sasa_slopes.items():
                descriptor_dict[f'all-temp-sasa_{key}_k={core_surface_k}_eq={eq_time}'] = slope
    
    # Parse conformational entropy from log files
    for temp in temps:
        log_file = f'sconf_{temp}.log'
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'Entropy' in line and 'J/mol K' in line:
                            if 'Schlitter' in line:
                                parts = line.split()
                                if len(parts) > 8:
                                    entropy = float(parts[8])
                                    descriptor_dict[f'sconf_schlitter_{temp}'] = entropy
                            elif 'Quasiharmonic' in line:
                                parts = line.split()
                                if len(parts) > 8:
                                    entropy = float(parts[8])
                                    descriptor_dict[f'sconf_quasiharmonic_{temp}'] = entropy
            except Exception as e:
                logger.warning(f"Failed to parse entropy log {log_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame([descriptor_dict])
    
    return df


def load_existing_descriptors(simulation_result: Dict, config: Dict) -> Dict:
    """
    Load existing descriptor computation results.
    
    Args:
        simulation_result: Dictionary containing simulation results
        config: Configuration dictionary
        
    Returns:
        Dictionary matching format from compute_descriptors
        
    Raises:
        FileNotFoundError: If descriptor file not found
    """
    logger.info("Loading existing descriptor computation results...")
    
    work_dir = Path(simulation_result["work_dir"]).resolve()
    
    # Try CSV first, then pickle
    descriptors_csv = work_dir / "descriptors.csv"
    descriptors_pkl = work_dir / "descriptors.pkl"
    
    descriptors_df = None
    
    if descriptors_csv.exists():
        try:
            descriptors_df = pd.read_csv(descriptors_csv)
            logger.info(f"Loaded descriptors from {descriptors_csv}")
        except Exception as e:
            logger.warning(f"Failed to load descriptors from CSV: {e}")
    
    if descriptors_df is None and descriptors_pkl.exists():
        try:
            import pickle
            with open(descriptors_pkl, 'rb') as f:
                descriptors_df = pickle.load(f)
            logger.info(f"Loaded descriptors from {descriptors_pkl}")
        except Exception as e:
            logger.warning(f"Failed to load descriptors from pickle: {e}")
    
    if descriptors_df is None:
        error_msg = f"Descriptor file not found when skipping descriptor computation.\n"
        error_msg += f"Expected one of:\n"
        error_msg += f"  - {descriptors_csv}\n"
        error_msg += f"  - {descriptors_pkl}\n"
        error_msg += f"\nWork directory: {work_dir}"
        raise FileNotFoundError(error_msg)
    
    # Get list of XVG files in work directory (if they exist)
    xvg_files = []
    try:
        xvg_files = [str(f.name) for f in work_dir.glob("*.xvg")]
        logger.info(f"Found {len(xvg_files)} XVG files in work directory")
    except Exception as e:
        logger.warning(f"Could not enumerate XVG files: {e}")
    
    result = {
        "status": "success",
        "descriptors_df": descriptors_df,
        "xvg_files": xvg_files,
        "work_dir": str(work_dir),
        "message": "Descriptor computation results loaded successfully"
    }
    
    logger.info(f"Successfully loaded descriptors. DataFrame shape: {descriptors_df.shape}")
    logger.info(f"Features: {len(descriptors_df.columns)}")
    
    return result


def _parse_xvg_file(xvg_file: str) -> Optional[np.ndarray]:
    """
    Parse GROMACS .xvg file and return data as numpy array.
    
    Args:
        xvg_file: Path to .xvg file
        
    Returns:
        Numpy array with data (time in first column, data in subsequent columns)
    """
    try:
        t, x, y, z, r = [], [], [], [], []
        
        with open(xvg_file, 'r') as f:
            for line in f:
                # Skip comments and metadata
                if line.startswith('#') or line.startswith('@'):
                    continue
                
                cols = line.split()
                
                if len(cols) == 0:
                    continue
                elif len(cols) == 2:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                elif len(cols) == 3:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                elif len(cols) == 4:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                    z.append(float(cols[3]))
                elif len(cols) == 5:
                    t.append(float(cols[0]))
                    r.append(float(cols[1]))
                    x.append(float(cols[2]))
                    y.append(float(cols[3]))
                    z.append(float(cols[4]))
        
        # Return appropriate array based on what was collected
        if len(r) > 0:
            return np.column_stack([r, x, y, z])
        elif len(z) > 0:
            return np.column_stack([x, y, z])
        elif len(y) > 0:
            return np.column_stack([x, y])
        elif len(x) > 0:
            return np.array(x)
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Failed to parse {xvg_file}: {e}")
        return None

