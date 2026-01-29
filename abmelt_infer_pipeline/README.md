#### Quick Commands

1. quick test - `python quick_test.py`
2. Run inference
    - Using pdb file - `python infer.py --pdb "/workspace/antibody-developability-abmelt/AbMelt/public_tm/train_pdbs/alemtuzumab.pdb" --name "alemtuzumab" --config configs/testing_config.yaml`
    - Using chains - `python infer.py --h [] --l [] --name [] --config configs/testing_config.yaml`

    `python infer.py --h "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS" --l "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIK" --name "alemtuzumab" --config configs/testing_config.yaml`



missing features because of eq_time being set to 0 in testing_config
- all-temp_lamda_b=25_eq=20
- all-temp-sasa_core_mean_k=20_eq=20
- all-temp-sasa_core_std_k=20_eq=20
- 