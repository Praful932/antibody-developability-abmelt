#!/usr/bin/env python3

"""
MD Simulation module for AbMelt inference pipeline.
Handles GROMACS preprocessing, system setup, and multi-temperature MD simulations.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add the original AbMelt src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "AbMelt" / "src"))

try:
    import gromacs
    from preprocess import protonation_state, canonical_index, edit_mdp
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

logger = logging.getLogger(__name__)


def run_md_simulation(structure_files: Dict[str, str], config: Dict) -> Dict[str, str]:
    """
    Run complete MD simulation workflow for antibody structure.
    
    Args:
        structure_files: Dictionary containing PDB file path and work directory
        config: Configuration dictionary with simulation parameters
        
    Returns:
        Dictionary containing trajectory files and simulation results
    """
    logger.info("Starting MD simulation workflow...")
    
    pdb_file = structure_files["pdb_file"]
    work_dir = Path(structure_files["work_dir"])
    antibody_name = work_dir.name
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    # Change to work directory
    os.chdir(work_dir)
    
    # Get the PDB filename (not full path) for GROMACS commands
    pdb_filepath = Path(pdb_file)
    pdb_filename = pdb_filepath.name

    
    try:
        # Setup GROMACS environment with MDP template path
        gromacs_config = config.get("gromacs", {})
        mdp_dir = gromacs_config.get("mdp_dir", "mdp")
        print(f"mdp_dir: {mdp_dir}")

        
        if not Path(mdp_dir).is_absolute():
            mdp_dir = str(Path(__file__).parent.parent / mdp_dir)
        
        setup_gromacs_environment(mdp_dir=mdp_dir)

        # Create temperature-specific MDP files in template directory
        temperatures = config.get("simulation")['temperatures']
        for temp in temperatures:
            temp_str = str(temp)
            for mdp_type in ["nvt", "npt", "md"]:
                mdp_file = Path(f"{mdp_type}_{temp_str}.mdp")
                src_file = Path(mdp_dir) / f"{mdp_type}.mdp"
                dst_file = mdp_file

                print(f"src_file: {src_file} , {src_file.exists()}")
                print(f"dst_file: {dst_file} , {dst_file.exists()}")
                
                if src_file.exists() and not dst_file.exists():
                    # Copy base MDP file and modify temperature in template directory
                    import shutil
                    shutil.copy2(src_file, dst_file)
                    _modify_mdp_temperature(dst_file, temp)
                    logger.info(f"Created {mdp_file} for {temp}K in template directory")
                elif dst_file.exists():
                    logger.info(f"Temperature-specific MDP file already exists: {mdp_file}")
                else:
                    logger.warning(f"Base MDP file not found: {src_file}")

        
        # Step 1: GROMACS preprocessing
        logger.info("Step 1: GROMACS preprocessing...")
        gromacs_files = _preprocess_for_gromacs(pdb_filename, pdb_filepath, config)
        
        # Step 2: System setup
        logger.info("Step 2: Setting up simulation system...")
        system_files = _setup_simulation_system(gromacs_files, config)
        
        # Step 3: Multi-temperature MD simulations
        logger.info("Step 3: Running multi-temperature MD simulations...")
        trajectory_files = _run_multi_temp_simulations(system_files, config)
        
        # Step 4: Process trajectories
        logger.info("Step 4: Processing trajectories...")
        processed_trajectories = _process_trajectories(trajectory_files, config)
        
        result = {
            "status": "success",
            "trajectory_files": processed_trajectories,
            "work_dir": str(work_dir),
            "message": "MD simulations completed successfully"
        }
        
        logger.info("MD simulation workflow completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"MD simulation failed: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def _preprocess_for_gromacs(pdb_filename: str, pdb_filepath, config: Dict) -> Dict[str, str]:
    """
    Preprocess PDB file for GROMACS: pKa calculation, conversion, and indexing.
    
    Args:
        pdb_filename: Name of input PDB file (should be in current working directory)
        config: Configuration dictionary
        
    Returns:
        Dictionary containing GROMACS files
    """
    logger.info("Preprocessing structure for GROMACS...")
    
    # Get simulation parameters
    sim_config = config["simulation"]
    pH = sim_config["pH"]
    force_field = sim_config["force_field"]
    water_model = sim_config["water_model"]
    
    # Step 1: Calculate protonation states
    logger.info("Calculating protonation states...")
    gromacs_input = protonation_state(
        pdb_filename=pdb_filename,
        pdb_path=pdb_filepath,
        pH=pH
    )

    
    # Step 2: Convert PDB to GROMACS format
    logger.info("Converting PDB to GROMACS format...")
    gromacs.pdb2gmx(
        f=pdb_filepath,
        o="processed.pdb",
        p="topol.top",
        ff=force_field,
        water=water_model,
        ignh=True,
        his=True,
        renum=True,
        input=gromacs_input
    )

    
    gromacs.pdb2gmx(
        f="processed.pdb",
        o="processed.gro",
        p="topol.top",
        ff=force_field,
        water=water_model,
        ignh=True,
        his=True,
        renum=True,
        input=gromacs_input
    )

    # Step 3: Create index groups for CDRs
    logger.info("Creating index groups for CDRs...")
    annotation = canonical_index(pdb=config['paths']['temp_dir'] / "processed.pdb")
    gromacs.make_ndx(f= config['paths']['temp_dir'] / "processed.gro", o=config['paths']['temp_dir'] / "index.ndx", input=annotation)

    
    return {
        "processed_pdb": config['paths']['temp_dir'] / "processed.pdb",
        "processed_gro": config['paths']['temp_dir'] / "processed.gro",
        "topology": config['paths']['temp_dir'] / "topol.top",
        "index": config['paths']['temp_dir'] / "index.ndx"
    }


def _setup_simulation_system(gromacs_files: Dict[str, str], config: Dict) -> Dict[str, str]:
    """
    Set up simulation system: box creation, solvation, and ion addition.
    
    Args:
        gromacs_files: Dictionary containing GROMACS files
        config: Configuration dictionary
        
    Returns:
        Dictionary containing system files
    """
    import os
    logger.info("Setting up simulation system...")
    
    # Get simulation parameters
    sim_config = config["simulation"]
    salt_concentration = sim_config["salt_concentration"]  # mM
    p_salt = sim_config["p_salt"]
    n_salt = sim_config["n_salt"]
    
    # Step 1: Create simulation box
    logger.info("Creating simulation box...")
    gromacs.editconf(
        f=gromacs_files["processed_gro"],
        o='box.gro',
        c=True,
        d='1.0',
        bt='triclinic'
    )   
    
    # Step 2: Add water
    logger.info("Adding water molecules...")
    gromacs.solvate(
        cp="box.gro",
        cs="spc216.gro",
        p=gromacs_files["topology"],
        o="solv.gro"
    )
    
    # Step 3: Add ions
    logger.info("Adding ions...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"GROMACS config.path: {gromacs.config.path}")
    logger.info(f"Looking for ions.mdp in: {[os.path.join(p, 'ions.mdp') for p in gromacs.config.path]}")

    
    try:
        ions = gromacs.config.get_templates('ions.mdp')
        logger.info(f"Found ions template: {ions}")
    except Exception as e:
        logger.error(f"Failed to get ions.mdp template: {e}")
        # Try to find the file manually
        for path in gromacs.config.path:
            ions_path = os.path.join(path, 'ions.mdp')
            if os.path.exists(ions_path):
                logger.info(f"Found ions.mdp manually at: {ions_path}")
                break
        raise


    
    gromacs.grompp(
        f=ions[0],
        c="solv.gro",
        p=gromacs_files["topology"],
        o="ions.tpr"
    )
    
    gromacs.genion(
        s="ions.tpr",
        o="solv_ions.gro",
        p=gromacs_files["topology"],
        pname=p_salt,
        nname=n_salt,
        conc=salt_concentration/1000,
        neutral=True,
        input=['13']
    )
    
    # Step 4: Energy minimization
    logger.info("Running energy minimization...")
    em = gromacs.config.get_templates('em.mdp')
    gromacs.grompp(
        f=em,
        c="solv_ions.gro",
        p=gromacs_files["topology"],
        o="em.tpr"
    )
    gromacs.mdrun(v=True, deffnm="em")

    
    return {
        "solvated_gro": config['paths']['temp_dir'] / "solv_ions.gro",
        "topology": gromacs_files["topology"],
        "index": gromacs_files["index"],
        "em_gro": config['paths']['temp_dir'] / "em.gro"
    }


def _run_multi_temp_simulations(system_files: Dict[str, str], config: Dict) -> Dict[str, str]:
    """
    Run multi-temperature MD simulations.
    
    Args:
        system_files: Dictionary containing system files
        config: Configuration dictionary
        
    Returns:
        Dictionary containing trajectory files
    """
    logger.info("Running multi-temperature MD simulations...")
    
    # Get simulation parameters
    sim_config = config["simulation"]
    temperatures = sim_config["temperatures"]
    simulation_time = sim_config["simulation_time"]  # ns
    gpu_enabled = sim_config["gpu_enabled"]
    
    # Get GROMACS settings
    gromacs_config = config["gromacs"]
    n_threads = gromacs_config["n_threads"]
    gpu_id = gromacs_config["gpu_id"]
    
    # Available pre-installed temperatures
    avail_temps = ['300', '310', '350', '373', '400']
    temps = [str(temp) for temp in temperatures]
    
    trajectory_files = {}

    for temp in temps:
        logger.info(f"Running simulation at {temp}K...")
        
        try:
            # Required temp files are already created
            # Use pre-installed temperature files
            trajectory_files[temp] = _run_preinstalled_temp_simulation(
                temp, system_files, simulation_time, gpu_enabled, n_threads, gpu_id
            )

            # if temp in avail_temps:
            
            # else:
            #     # Create custom temperature files
            #     trajectory_files[temp] = _run_custom_temp_simulation(
            #         temp, system_files, simulation_time, gpu_enabled, n_threads, gpu_id
            #     )
                
        except Exception as e:
            logger.error(f"Simulation failed at {temp}K: {e}")
            raise
    
    return trajectory_files


def _run_preinstalled_temp_simulation(temp: str, system_files: Dict[str, str], 
                                    simulation_time: int, gpu_enabled: bool,
                                    n_threads: int, gpu_id: int) -> Dict[str, str]:
    """Run simulation using pre-installed temperature files."""

    # TODO : start here
    
    # Get MDP templates
    nvt = gromacs.config.get_templates('nvt_' + temp + '.mdp')
    npt = gromacs.config.get_templates('npt_' + temp + '.mdp')
    md = gromacs.config.get_templates('md_' + temp + '.mdp')
    
    # NVT equilibration
    logger.info(f"Running NVT equilibration at {temp}K...")
    gromacs.grompp(
        f=nvt[0],
        o='nvt_' + temp + '.tpr',
        c=system_files["em_gro"],
        r=system_files["em_gro"],
        p=system_files["topology"]
    )
    
    if gpu_enabled:
        gromacs.mdrun(
            deffnm='nvt_' + temp,
            ntomp=str(n_threads),
            nb='gpu',
            pme='gpu',
            update='gpu',
            bonded='cpu',
            pin='on'
        )
    else:
        gromacs.mdrun(deffnm='nvt_' + temp, ntomp=str(n_threads))
    
    # NPT equilibration
    logger.info(f"Running NPT equilibration at {temp}K...")
    gromacs.grompp(
        f=npt[0],
        o='npt_' + temp + '.tpr',
        t='nvt_' + temp + '.cpt',
        c='nvt_' + temp + '.gro',
        r='nvt_' + temp + '.gro',
        p=system_files["topology"],
        maxwarn='1'
    )
    
    if gpu_enabled:
        gromacs.mdrun(
            deffnm='npt_' + temp,
            ntomp=str(n_threads),
            nb='gpu',
            pme='gpu',
            update='gpu',
            bonded='cpu',
            pin='on'
        )
    else:
        gromacs.mdrun(deffnm='npt_' + temp, ntomp=str(n_threads))
    
    # Production MD
    logger.info(f"Running production MD at {temp}K...")
    if simulation_time == 100:
        gromacs.grompp(
            f=md[0],
            o='md_' + temp + '.tpr',
            t='npt_' + temp + '.cpt',
            c='npt_' + temp + '.gro',
            p=system_files["topology"]
        )
        
        if gpu_enabled:
            gromacs.mdrun(
                deffnm='md_' + temp,
                ntomp=str(n_threads),
                nb='gpu',
                pme='gpu',
                update='gpu',
                bonded='cpu',
                pin='on'
            )
        else:
            gromacs.mdrun(deffnm='md_' + temp, ntomp=str(n_threads))
    else:
        # Custom simulation time
        new_mdp = 'md_' + temp + '_' + str(simulation_time) + '.mdp'
        template_path = Path(gromacs.config.get_templates('md_' + temp + '.mdp')[0])
        _modify_mdp_nsteps(template_path, new_mdp, int(simulation_time*1000*1000/2))
        new_md = gromacs.config.get_templates(new_mdp)
        
        gromacs.grompp(
            f=new_md[0],
            o='md_' + temp + '_' + str(simulation_time) + '.tpr',
            t='npt_' + temp + '.cpt',
            c='npt_' + temp + '.gro',
            p=system_files["topology"]
        )
        
        if gpu_enabled:
            gromacs.mdrun(
                deffnm='md_' + temp + '_' + str(simulation_time),
                ntomp=str(n_threads),
                nb='gpu',
                pme='gpu',
                update='cpu',
                # update='gpu',
                bonded='cpu',
                pin='on'
            )
        else:
            gromacs.mdrun(deffnm='md_' + temp + '_' + str(simulation_time), ntomp=str(n_threads))
    
    return {
        "tpr_file": f"md_{temp}.tpr",
        "xtc_file": f"md_{temp}.xtc",
        "gro_file": f"md_{temp}.gro",
        "log_file": f"md_{temp}.log"
    }


def _run_custom_temp_simulation(temp: str, system_files: Dict[str, str],
                              simulation_time: int, gpu_enabled: bool,
                              n_threads: int, gpu_id: int) -> Dict[str, str]:
    """Run simulation with custom temperature by modifying MDP files."""    
    
    # Create custom MDP files
    nvt_mdp = 'nvt_' + temp + '.mdp'
    npt_mdp = 'npt_' + temp + '.mdp'
    md_mdp = 'md_' + temp + '.mdp'
    
    # Modify temperature in MDP files
    edit_mdp('nvt_300.mdp', new_mdp=nvt_mdp, ref_t=[temp, temp], gen_temp=temp)
    edit_mdp('npt_300.mdp', new_mdp=npt_mdp, ref_t=[temp, temp])
    edit_mdp('md_300.mdp', new_mdp=md_mdp, ref_t=[temp, temp])
    
    # Get templates
    new_nvt = gromacs.config.get_templates(nvt_mdp)
    new_npt = gromacs.config.get_templates(npt_mdp)
    new_md = gromacs.config.get_templates(md_mdp)
    
    # Run simulations (similar to preinstalled but with custom MDPs)
    # NVT
    gromacs.grompp(
        f=new_nvt[0],
        o='nvt_' + temp + '.tpr',
        c=system_files["em_gro"],
        r=system_files["em_gro"],
        p=system_files["topology"]
    )
    
    if gpu_enabled:
        gromacs.mdrun(
            deffnm='nvt_' + temp,
            ntomp=str(n_threads),
            nb='gpu',
            pme='gpu',
            update='gpu',
            bonded='cpu',
            pin='on'
        )
    else:
        gromacs.mdrun(deffnm='nvt_' + temp, ntomp=str(n_threads))
    
    # NPT
    gromacs.grompp(
        f=new_npt[0],
        o='npt_' + temp + '.tpr',
        t='nvt_' + temp + '.cpt',
        c='nvt_' + temp + '.gro',
        r='nvt_' + temp + '.gro',
        p=system_files["topology"],
        maxwarn='1'
    )
    
    if gpu_enabled:
        gromacs.mdrun(
            deffnm='npt_' + temp,
            ntomp=str(n_threads),
            nb='gpu',
            pme='gpu',
            update='gpu',
            bonded='cpu',
            pin='on'
        )
    else:
        gromacs.mdrun(deffnm='npt_' + temp, ntomp=str(n_threads))
    
    # Production MD
    if simulation_time == 100:
        gromacs.grompp(
            f=new_md[0],
            o='md_' + temp + '.tpr',
            t='npt_' + temp + '.cpt',
            c='npt_' + temp + '.gro',
            p=system_files["topology"]
        )
        
        if gpu_enabled:
            gromacs.mdrun(
                deffnm='md_' + temp,
                ntomp=str(n_threads),
                nb='gpu',
                pme='gpu',
                update='gpu',
                bonded='cpu',
                pin='on'
            )
        else:
            gromacs.mdrun(deffnm='md_' + temp, ntomp=str(n_threads))
    else:
        # Custom simulation time
        new_mdp = 'md_' + temp + '_' + str(simulation_time) + '.mdp'
        template_path = Path(gromacs.config.get_templates(md_mdp)[0])
        _modify_mdp_nsteps(template_path, new_mdp, int(simulation_time*1000*1000/2))
        new_md_custom = gromacs.config.get_templates(new_mdp)
        
        gromacs.grompp(
            f=new_md_custom[0],
            o='md_' + temp + '_' + str(simulation_time) + '.tpr',
            t='npt_' + temp + '.cpt',
            c='npt_' + temp + '.gro',
            p=system_files["topology"]
        )
        
        if gpu_enabled:
            gromacs.mdrun(
                deffnm='md_' + temp + '_' + str(simulation_time),
                ntomp=str(n_threads),
                nb='gpu',
                pme='gpu',
                update='gpu',
                bonded='cpu',
                pin='on'
            )
        else:
            gromacs.mdrun(deffnm='md_' + temp + '_' + str(simulation_time), ntomp=str(n_threads))
    
    return {
        "tpr_file": f"md_{temp}.tpr",
        "xtc_file": f"md_{temp}.xtc",
        "gro_file": f"md_{temp}.gro",
        "log_file": f"md_{temp}.log"
    }


def _process_trajectories(trajectory_files: Dict[str, Dict[str, str]], config: Dict) -> Dict[str, Dict[str, str]]:
    """
    Process trajectories: remove PBC and prepare for analysis.
    
    Args:
        trajectory_files: Dictionary containing trajectory files for each temperature
        config: Configuration dictionary
        
    Returns:
        Dictionary containing processed trajectory files
    """
    logger.info("Processing trajectories...")
    
    sim_config = config.get("simulation", {})
    simulation_time = sim_config.get("simulation_time", 100)
    
    processed_trajectories = {}
    
    for temp, files in trajectory_files.items():
        logger.info(f"Processing trajectory at {temp}K...")
        
        try:
            if simulation_time == 100:
                # Remove periodic boundary conditions
                gromacs.trjconv(
                    f=files["xtc_file"],
                    s=files["tpr_file"],
                    pbc='whole',
                    o='md_whole_' + temp + '.xtc',
                    input=['0']
                )
                
                gromacs.trjconv(
                    f='md_whole_' + temp + '.xtc',
                    s=files["tpr_file"],
                    pbc='nojump',
                    o='md_nopbcjump_' + temp + '.xtc',
                    input=['1']
                )
                
                # Create final trajectory and reference structure
                gromacs.trjconv(
                    f='md_nopbcjump_' + temp + '.xtc',
                    s=files["tpr_file"],
                    b='0',
                    e='0',
                    o='md_final_' + temp + '.gro',
                    input=['1']
                )
                
                gromacs.trjconv(
                    f='md_nopbcjump_' + temp + '.xtc',
                    s=files["tpr_file"],
                    dt='0',
                    o='md_final_' + temp + '.xtc',
                    input=['1']
                )
            else:
                # Custom simulation time
                xtc_file = f"md_{temp}_{simulation_time}.xtc"
                tpr_file = f"md_{temp}_{simulation_time}.tpr"
                
                gromacs.trjconv(
                    f=xtc_file,
                    s=tpr_file,
                    pbc='whole',
                    o='md_whole_' + temp + '_' + str(simulation_time) + '.xtc',
                    input=['0']
                )
                
                gromacs.trjconv(
                    f='md_whole_' + temp + '_' + str(simulation_time) + '.xtc',
                    s=tpr_file,
                    pbc='nojump',
                    o='md_nopbcjump_' + temp + '_' + str(simulation_time) + '.xtc',
                    input=['1']
                )
                
                gromacs.trjconv(
                    f='md_nopbcjump_' + temp + '_' + str(simulation_time) + '.xtc',
                    s=tpr_file,
                    b='0',
                    e='0',
                    o='md_final_' + temp + '.gro',
                    input=['1']
                )
                
                gromacs.trjconv(
                    f='md_nopbcjump_' + temp + '_' + str(simulation_time) + '.xtc',
                    s=tpr_file,
                    dt='0',
                    o='md_final_' + temp + '.xtc',
                    input=['1']
                )
                
                # Rename for consistency
                os.rename(tpr_file, files["tpr_file"])
            
            processed_trajectories[temp] = {
                "final_xtc": f'md_final_{temp}.xtc',
                "final_gro": f'md_final_{temp}.gro',
                "tpr_file": files["tpr_file"],
                "log_file": files["log_file"]
            }
            
        except Exception as e:
            logger.error(f"Trajectory processing failed at {temp}K: {e}")
            raise
    
    return processed_trajectories


def _modify_mdp_temperature(mdp_file: Path, temperature: int):
    """
    Modify MDP file to set the correct temperature.
    
    Args:
        mdp_file: Path to MDP file to modify
        temperature: Target temperature in Kelvin
    """
    try:
        # Read the MDP file
        with open(mdp_file, 'r') as f:
            lines = f.readlines()
        
        # Modify temperature-related lines
        modified_lines = []
        for line in lines:
            if line.strip().startswith('ref_t'):
                # Update reference temperature
                modified_lines.append(f"ref_t                   = {temperature}   {temperature}                     ; reference temperature, one for each group, in K\n")
            elif line.strip().startswith('gen_temp'):
                # Update generation temperature
                modified_lines.append(f"gen_temp                = {temperature}       ; temperature for Maxwell distribution\n")
            else:
                modified_lines.append(line)
        
        # Write the modified MDP file
        with open(mdp_file, 'w') as f:
            f.writelines(modified_lines)
            
    except Exception as e:
        logger.error(f"Failed to modify MDP file {mdp_file}: {e}")


def _modify_mdp_nsteps(template_file: Path, output_name: str, nsteps: int):
    """
    Copy an MDP template into the working directory and update its nsteps value.
    """
    try:
        from shutil import copy2

        destination = Path(output_name)
        copy2(template_file, destination)

        with open(destination, 'r') as src:
            lines = src.readlines()

        updated_lines = []
        nsteps_updated = False
        for line in lines:
            if line.strip().startswith('nsteps'):
                updated_lines.append(f"nsteps                  = {nsteps}\n")
                nsteps_updated = True
            else:
                updated_lines.append(line)

        if not nsteps_updated:
            updated_lines.append(f"\n; inserted by pipeline\nnsteps                  = {nsteps}\n")

        with open(destination, 'w') as dst:
            dst.writelines(updated_lines)

        gromacs.config.templates[output_name] = str(destination.resolve())
        logger.info(f"Created {output_name} with nsteps={nsteps}")
    except Exception as e:
        logger.error(f"Failed to update nsteps for {template_file}: {e}")
        raise


def setup_gromacs_environment(gromacs_path: str = None, mdp_dir: str = None):
    """
    Setup GROMACS environment and configuration.
    
    Args:
        gromacs_path: Path to GROMACS installation (optional)
        mdp_dir: Path to MDP template files directory (optional)
    """
    if gromacs_path:
        gromacs.config.set_gmxrc_environment(gromacs_path)
    
    # Initialize GROMACS configuration first
    gromacs.config.get_configuration()
    gromacs.tools.registry
    gromacs.config.check_setup()
    gromacs.config.setup()
    
    # Set MDP template directory path AFTER GROMACS initialization
    if mdp_dir:
        gromacs.config.path.append(mdp_dir)
        logger.info(f"Added GROMACS template path : {mdp_dir}")
        for f in Path(mdp_dir).glob("*.mdp"):
            name = f.name
            if f.name not in gromacs.config.templates:
                gromacs.config.templates[name] = str(f)
        print(f"gromacs.config.templates: {gromacs.config.templates}")
        logger.info(f"Current gromacs.config.path: {gromacs.config.path}")
        
        # Check if template files exist
        import os
        for template_file in ['ions.mdp', 'em.mdp', 'nvt.mdp', 'npt.mdp', 'md.mdp']:
            template_path = os.path.join(mdp_dir, template_file)
            exists = os.path.exists(template_path)
            logger.info(f"Template file {template_file}: {'EXISTS' if exists else 'NOT FOUND'} at {template_path}")
    
    logger.info(f"GROMACS version: {gromacs.release()}")



def validate_simulation_setup(config: Dict) -> bool:
    """
    Validate that all required components are available for MD simulation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if setup is valid, False otherwise
    """
    try:
        # Check GROMACS installation
        gromacs.config.check_setup()
        
        # Check required MDP files
        required_mdp_files = ['em.mdp', 'ions.mdp']
        for temp in config.get("simulation", {}).get("temperatures", [300, 350, 400]):
            temp_str = str(temp)
            if temp_str in ['300', '310', '350', '373', '400']:
                required_mdp_files.extend([
                    f'nvt_{temp_str}.mdp',
                    f'npt_{temp_str}.mdp',
                    f'md_{temp_str}.mdp'
                ])
        
        # Check if MDP files exist (this would need to be implemented)
        # For now, just return True
        return True
        
    except Exception as e:
        logger.error(f"Simulation setup validation failed: {e}")
        return False
