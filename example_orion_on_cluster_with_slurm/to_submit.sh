#!/bin/bash
#SBATCH --array=2
#SBATCH --job-name=orion_test
#SBATCH --output=other_logs/out_%a.log
#SBATCH --error=other_logs/err_%a.log
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=1Gb

module load python/3.6
source ../../ve/bin/activate
orion -v hunt --config orion_config.yaml ../my_exp_code/main.py --config exp_config.yaml --saved_model '{exp.working_dir}/{exp.name}_{trial.id}/model.pt' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'
