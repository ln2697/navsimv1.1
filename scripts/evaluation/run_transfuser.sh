TRAIN_TEST_SPLIT=navtest
CHECKPOINT=outputs/training/800_navsim_baseline/001_posttrain1_0/251107_091149

export NAVSIM_DEVKIT_ROOT="${PROJECT_DIR}/3rd_party/navsim_workspace/navsimv1.1"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval 
