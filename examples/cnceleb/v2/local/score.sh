#!/bin/bash
# coding:utf-8
# Author: Chengdong Liang

exp_dir=
trials="CNC-Eval-Core.lst"

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=data/eval/trials
  for x in $trials; do
    echo $x
    python wespeaker/bin/score.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/eval/xvector.scp \
      --cal_mean True \
      --cal_mean_dir ${exp_dir}/embeddings/cnceleb_train \
      ${trials_dir}/${x}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.score \
        2>&1 | tee -a ${scores_dir}/cnc_cos_result

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.score
  done
fi