#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --output=out/testing_%A_%a.out
#SBATCH --error=err/testing_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=11:59:59
#SBATCH --mem=25G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=rrg-mbowling-ad
#SBATCH --mail-user=kapeluck@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# module load python/3.11 StdEnv/2023 gcc opencv/4.8.1 swig
# virtualenv pyenv
# source pyenv/bin/activate
# pip install 'requests[socks]' --no-index
# pip install --no-cache-dir autorom gymnasium "gymnasium[classic-control,box2d,atari,other]" "numpy<2" "stable_baselines3==2.0.0a1" tqdm tyro torch tensorboard wandb --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu && \
# AutoROM -y --install-dir atari_roms


module load StdEnv/2020
module load python/3.9
module load gcc cuda/11.1.1 cudnn/8.2.0 #opencv swig
module load snappy
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip==22.1.2
pip install --no-index -r requirements_cc.txt


python -m dqn_zoo.gym_atari_test
python -m dqn_zoo.networks_test # 1 failure
python -m dqn_zoo.parts_test
python -m dqn_zoo.replay_test # failure due to python-snappy import not working
python -m dqn_zoo.c51.run_atari_test # failure due to python-snappy import not working
python -m dqn_zoo.double_q.run_atari_test # same as above and onwards
python -m dqn_zoo.dqn.run_atari_test
python -m dqn_zoo.iqn.run_atari_test
python -m dqn_zoo.prioritized.run_atari_test
python -m dqn_zoo.qrdqn.run_atari_test
python -m dqn_zoo.rainbow.run_atari_test

# pip install "requests[socks]" --no-index
# pip install --no-cache-dir autorom
# mkdir -p $SLURM_TMPDIR/ROMS
# pip install --no-index ale-py
# AutoROM --accept-license && AutoROM --install-dir $SLURMTMP_DIR/ROMS
# ale-import-roms $SLURM_TMPDIR/ROMS
# module load gcc cuda cudnn opencv swig
# pip install --no-index .
# module load gcc/9.3.0 cuda/11.8 cudnn/8.6
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64;$EBROOTCUDNN/lib"

#  the dqn train file
# python dopamine/discrete_domains/train.py --base_dir test --gin_files dopamine/jax/agents/dqn/configs/dqn.gin
# python tests/dopamine/atari_init_test.py

# mkdir -p ./logs/$SLURM_JOB_NAME

# python example_acrobot.py \
#     --logdir="./logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}" \
#     --seed="${SLURM_ARRAY_TASK_ID}"

# touch ./logs/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/complete