rm -fr mlruns
rm -fr my_working_dir
orion -v hunt --config orion_config.yaml ../my_exp_code/main.py --config exp_config.yaml --saved_model '{exp.working_dir}/{exp.name}_{trial.id}/model.pt'
