#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --partition=2080-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=200gb

eval "$(conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="navsimv1.1" # Check if CONDA_INTERPRETER is not set, then set it to navsimv1.1
fi
source activate "$CONDA_INTERPRETER"

python "${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1/navsim/planning/script/run_dataset_caching.py" \
    agent=carla_transfuser_agent \
    train_test_split=navtrain \
    experiment_name=navtrain_cache_transfuser \
    cache_path="${LEAD_PROJECT_ROOT}/data/navsim_training_cache/navtrain"