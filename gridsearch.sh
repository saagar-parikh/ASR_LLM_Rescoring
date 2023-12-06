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


# for lambda in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}
for lambda in {0.91,0.92,0.93,0.94,0.96,0.97,1.00}
do
    echo "lambda: " $lambda
    python combined_scores.py --test_set test_other --lambda $lambda
    python compute_error_rate.py --test_set test_other --lambda $lambda
done

# python llm_scoring.py --test_set test_other
# python combined_scores.py --test_set test_other