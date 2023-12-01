#!/bin/bash
cd /ocean/projects/cis230078p/dannemil/espnet_tutorial/espnet/egs2/librispeech_100/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.asr_inference --batch_size 1 --ngpu 1 --data_path_and_name_and_type dump/raw/dev_other/wav.scp,speech,kaldi_ark --key_file exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/keys.${SLURM_ARRAY_TASK_ID}.scp --asr_train_config exp/asr_train_asr_raw_en_bpe5000_sp/config.yaml --asr_model_file exp/asr_train_asr_raw_en_bpe5000_sp/valid.acc.ave.pth --output_dir exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/output.${SLURM_ARRAY_TASK_ID} --config conf/decode_asr.yaml 
EOF
) >exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.asr_inference --batch_size 1 --ngpu 1 --data_path_and_name_and_type dump/raw/dev_other/wav.scp,speech,kaldi_ark --key_file exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/keys.${SLURM_ARRAY_TASK_ID}.scp --asr_train_config exp/asr_train_asr_raw_en_bpe5000_sp/config.yaml --asr_model_file exp/asr_train_asr_raw_en_bpe5000_sp/valid.acc.ave.pth --output_dir exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/output.${SLURM_ARRAY_TASK_ID} --config conf/decode_asr.yaml  ) &>>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/asr_inference.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/done.2557767.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --time 48:00:00 -p GPU-shared --gres=gpu:1 -c 1  --open-mode=append -e exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/asr_inference.log -o exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/asr_inference.log --array 1-8 /ocean/projects/cis230078p/dannemil/espnet_tutorial/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/asr_inference.sh >>exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/dev_other/logdir/q/asr_inference.log 2>&1
