#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --array=0-4%5

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}


# TASK_NAME=mnli
TASK_NAME=sst2
metric="accuracy"
# wandb env variables
export WANDB_PROJECT=momentum.adapter.adam
export WANDB_WATCH="false"

DATE=`date +%Y%m%d%h`

# declare -a root_seed_list=(42 2 4 6 8)
# seed=${root_seed_list[$SLURM_ARRAY_TASK_ID]}
m_step_size=$1
beta_2=$2
s_step_size=$3
adapter_option=$4

case ${adapter_option} in
    sequential)
        attn_mode="adapter"
        attn_option="sequential"
        attn_composition="add"
        attn_bn=200  # attn bottleneck dim

        ffn_mode="adapter"
        ffn_option="sequential"
        ffn_adapter_layernorm_option="none"
        ffn_adapter_init_option="bert"
        ffn_adapter_scalar="1"
        ffn_bn=200 # ffn bottleneck dim
    ;;
    parallel)
        attn_mode="prefix"
        attn_option="concat"
        attn_composition="add"
        attn_bn=30  # attn bottleneck dim

        ffn_mode="adapter"
        ffn_option="parallel"
        ffn_adapter_layernorm_option="none"
        ffn_adapter_init_option="lora"
        ffn_adapter_scalar="4"
        ffn_bn=512 # ffn bottleneck dim
    ;;
esac

# declare -a seed_list=(42)
declare -a seed_list=(42)
# declare -a seed_list=(8)
# declare -a seed_list=(6 8)
# declare -a seed_list=(${root_seed})
# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi

# set to 1 for debug mode which only
# uses 1600 training examples
debug=0

# set to "wandb" to use weights & bias
report_to="wandb"

bsz=16
gradient_steps=1

# lr=5e-5
lr=1e-4
max_grad_norm=1
# lr=1e-5
# weight_decay=0
weight_decay=0.1
warmup_updates=0
warmup_ratio=0.06
max_steps=-1
num_train_epochs=10
max_tokens_per_batch=0
max_seq_length=512

lr_scheduler_type="polynomial"
#metric=bleu
unfreeze='ef_'
max_eval_samples=1600
logging_steps=50

eval_strategy="epoch"
save_steps=5000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    weight_decay=0
    max_grad_norm=1
    max_train_samples=1000
    max_eval_samples=150
    bsz=10
    gradient_steps=1
    num_train_epochs=5
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi

for seed in "${seed_list[@]}"; do
exp_name=glue.${TASK_NAME}.am_${attn_mode}.ao_${attn_option}
exp_name+=.unfrz_${unfreeze}.ne${num_train_epochs}
exp_name+=.warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}
exp_name+=.m_${m_step_size}.beta2_${beta_2}.s_${s_step_size}
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
echo "${SAVE}"
rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock

# python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/text-classification/run_glue.py \
# --max_eval_samples ${max_eval_samples} \
CUDA_VISIBLE_DEVICES=0,1 python -u tasks/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --seed ${seed} \
    --unfreeze_params ${unfreeze} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --max_seq_length ${max_seq_length} \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${TASK_NAME}.${DATE}.${exp_name}.${adapter_option}.$5 \
    --overwrite_output_dir \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --ddp_find_unused_parameter "False" \
    --m_step_size ${m_step_size} \
    --s_step_size ${s_step_size} \
    --mask_option $5 \
    --beta1 ${m_step_size} \
    --beta2 ${beta_2} \
    --epsilon 1e-16 \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
done

