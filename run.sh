#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:1


#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /ocean/projects/cis230078p/sparikh1/espnet/tools/miniconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
cd /ocean/projects/cis230078p/sparikh1/project/ASR_LLM_Rescoring

#run pre-compiled program which is already in your project space

python llm_scoring.py --test_set test_other
# python combined_scores.py --test_set test_other