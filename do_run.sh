#!/bin/bash
#BSUB -q gpuv100          # o la cola GPU que uses (puede ser gputitan, gpuv100, etc.)
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -W 02:00             # tiempo máximo
#BSUB -R "rusage[mem=8GB]"
#BSUB -J wd_sweep
#BSUB -o logs/SGD_momentum_Nesterov_%J.out
#BSUB -e logs/SGD_momentum_Nesterov_%J.err

# Activa tu entorno
source /dtu/blackhole/0b/213963/venvs/dl_project/bin/activate

# Asegúrate de estar en el directorio del proyecto
cd /dtu/blackhole/0b/213963/dl_project/gnn_intro

# Asegúrate de que Hydra guarde cada run en un directorio único
export HYDRA_FULL_ERROR=1

python src/run.py logger.name="GIN" 