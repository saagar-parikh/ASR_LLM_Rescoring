#!/bin/bash
cd /ocean/projects/cis230078p/dannemil/espnet_tutorial/espnet/egs2/librispeech_100/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
pyscripts/utils/calculate_rtf.py --log-dir exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir --log-name asr_inference --input-shift 0.0625 --start-times-marker "speech length" --end-times-marker "best hypo" --inf-num 1 
EOF
) >exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( pyscripts/utils/calculate_rtf.py --log-dir exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir --log-name asr_inference --input-shift 0.0625 --start-times-marker "speech length" --end-times-marker "best hypo" --inf-num 1  ) &>>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
echo '#' Accounting: end_time=$time2 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
echo '#' Finished at `date` with status $ret >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/calculate_rtf.log
[ $ret -eq 137 ] && exit 100;
touch exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/done.2884039.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --time 48:00:00 -p RM-shared  --open-mode=append -e exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/calculate_rtf.log -o exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/calculate_rtf.log --array 1-1 /ocean/projects/cis230078p/dannemil/espnet_tutorial/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/calculate_rtf.sh >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/calculate_rtf.log 2>&1
