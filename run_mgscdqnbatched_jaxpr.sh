#!/bin/bash
#SBATCH --job-name=mgscdqn_batched_jaxpr_plotting
#SBATCH --output=/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_jaxpr_plotting/mgscdqn_batched_jaxpr_plotting_%A_%a.out
#SBATCH --error=/home/kapeluck/scratch/dqn_zoo_results/results/mgscdqn_batched_jaxpr_plotting/mgscdqn_batched_jaxpr_plotting_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=7:59:59
#SBATCH --mem=25G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=rrg-mbowling-ad
#SBATCH --mail-user=kapeluck@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# To download the roms:
# module load python/3.11 StdEnv/2023 gcc opencv/4.8.1 swig
# virtualenv pyenv
# source pyenv/bin/activate
# pip install 'requests[socks]' --no-index
# pip install --no-cache-dir autorom gymnasium "gymnasium[classic-control,box2d,atari,other]" "numpy<2" "stable_baselines3==2.0.0a1" tqdm tyro torch tensorboard wandb --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu && \
# AutoROM -y --install-dir atari_roms


module load StdEnv/2020
module load python/3.9
module load gcc/9.3.0 cuda/11.2.2 cudnn/8.2
module load snappy
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip==22.1.2
pip install --no-index tensorflow==2.9.0
# pip wheel python-snappy==0.6.1
# pip wheel gym==0.19.0
# pip wheel tensorflow-probability==0.17.0
pip install --no-index ~/python_snappy-0.6.1-cp39-cp39-linux_x86_64.whl
pip install --no-index ~/gym-0.19.0-py3-none-any.whl
pip install --no-index ~/tensorflow_probability-0.17.0-py2.py3-none-any.whl
pip install --no-index -r requirements_cc.txt
python -m atari_py.import_roms atari_roms

pip install --no-index graphviz # for jaxpr viewing
python -m dqn_zoo.dqn_mgsc_batched.jaxpr_viewing_atari --meta_batch_size=10