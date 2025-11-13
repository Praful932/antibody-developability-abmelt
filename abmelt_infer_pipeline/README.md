#### Quick Commands

1. quick test - `python quick_test.py`
2. Run inference
    - Using pdb file - `python infer.py --pdb "/workspace/antibody-developability-abmelt/AbMelt/public_tm/train_pdbs/alemtuzumab.pdb" --name "alemtuzumab" --config configs/testing_config.yaml`
    - Using chains - `python infer.py --h [] --l [] --name [] --config configs/testing_config.yaml`

    `python infer.py --h "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS" --l "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIK" --name "alemtuzumab" --config configs/testing_config.yaml`