#!/bin/bash --login


# HOW TO USE:
# 1. Edit the variables in the config section below.
# 2. Submit the job using: sbatch run.slurm


#SBATCH --job-name=single_moe_run
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --partition=gpuA
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00



ABLATION_NAME="Disjoint_BCE_Weighted_CED"

# Choices: "disjoint_moe", "early_fusion_moe"
MODEL_ARCHITECTURE="disjoint_moe"
# Choices: "bce", "focal"
LOSS_FN="bce"
# Choices: "sum", "weighted" (for disjoint_moe) or "concat", "weighted" (for early_fusion_moe)
FUSION_STRATEGY="weighted"
BATCH_SIZE=8
MAIN_LR="1e-5"
NUM_EPOCHS=50
PATIENCE=7
TOP_K=2


# To DISABLE a flag, set the variable to an empty string ("").
FLAG_USE_STABILIZED_EXPERT="--use_stabilized_expert"
FLAG_USE_RESIDUAL_TEXT_BLOCK="" 
FLAG_USE_MISSING_EMBEDS=""

# --- Complete Expert Dropout (CED) Parameters ---
EXPERT_DROPOUT_MIN_K=1
EXPERT_DROPOUT_MAX_K=2
# Persistence probability
EXPERT_DROPOUT_PERSIST_PROB=0.3
# The epochs during which to apply dropout.
EXPERT_DROPOUT_START_EPOCH=1
EXPERT_DROPOUT_END_EPOCH=40

# --- Other MoE Parameters ---
# Amount of noise to add to gating logits for exploration.
MOE_NOISE_EPS=0.3
# The coefficient for the MoE load-balancing loss.
MOE_LOSS_COEF=0.000





BASE_SCRATCH_DIR="/net/scratch/u75430mk/mimic_iii_scratch"
ABLATION_DIR="${BASE_SCRATCH_DIR}/ablations"
OUTPUT_DIR="${ABLATION_DIR}/${ABLATION_NAME}"
RESULTS_FILE="${ABLATION_DIR}/master_results.csv"

cd "$BASE_SCRATCH_DIR"
mkdir -p slurm
mkdir -p "$OUTPUT_DIR"
echo "WORKING DIR: $(pwd)"

module purge
module load apps/binapps/anaconda3/2023.03
module load cuda/12.6.2
source activate moe_env
export TOKENIZERS_PARALLELISM=false


# --- Print Configuration for Logging ---


echo "Model Architecture: ${MODEL_ARCHITECTURE}"
echo "Loss Function:      ${LOSS_FN}"
echo "Fusion Strategy:    ${FUSION_STRATEGY}"
echo "Learning Rate:      ${MAIN_LR}"
echo "Batch Size:         ${BATCH_SIZE}"
echo "Top-K Routing:      ${TOP_K}"
echo "Expert Dropout:     K=${EXPERT_DROPOUT_MIN_K}-${EXPERT_DROPOUT_MAX_K}, Persist=${EXPERT_DROPOUT_PERSIST_PROB}"
echo "Stabilized Expert:  ${FLAG_USE_STABILIZED_EXPERT:-Disabled}"
echo "Missing Embeds:     ${FLAG_USE_MISSING_EMBEDS:-Disabled}"



# --- Assemble and Run the Final Command ---
/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/u75430mk/.conda/envs/moe_env/bin/accelerate launch run_training.py \
  --file_path "${BASE_SCRATCH_DIR}/data/mimic_iii" \
  --output_dir "$OUTPUT_DIR" \
  --results_file "$RESULTS_FILE" \
  --seed 42 \
  --model_architecture "$MODEL_ARCHITECTURE" \
  --loss_function "$LOSS_FN" \
  --fusion_strategy "$FUSION_STRATEGY" \
  --early_fusion_strategy "$FUSION_STRATEGY" \
  $FLAG_USE_STABILIZED_EXPERT \
  $FLAG_USE_RESIDUAL_TEXT_BLOCK \
  $FLAG_USE_MISSING_EMBEDS \
  --top_k "$TOP_K" \
  --expert_dropout_min_k "$EXPERT_DROPOUT_MIN_K" \
  --expert_dropout_max_k "$EXPERT_DROPOUT_MAX_K" \
  --expert_dropout_persistence_prob "$EXPERT_DROPOUT_PERSIST_PROB" \
  --expert_dropout_epoch_start "$EXPERT_DROPOUT_START_EPOCH" \
  --expert_dropout_epoch_end "$EXPERT_DROPOUT_END_EPOCH" \
  --moe_noise_eps "$MOE_NOISE_EPS" \
  --moe_loss_coef "$MOE_LOSS_COEF" \
  --num_experts 6 8 8 \
  --batch_size "$BATCH_SIZE" \
  --main_lr "$MAIN_LR" \
  --weight_decay 0.06 \
  --patience "$PATIENCE" \
  --num_train_epochs "$NUM_EPOCHS" \
  --ts_time_embedder "time2vec" \
  --ts_attention_type "irregular_aware"

echo
echo "JOB END TIME: $(date)"
