MVS_TRAINING="/"  # path to plant dataset
CKPT=".ckpt" # path to checkpoint
LOG_DIR="outputs/plants_finetune"
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi


NGPUS=1
BATCH_SIZE=1
python finetune.py \
--logdir=$LOG_DIR \
--dataset=plant_models \
--trainpath=$MVS_TRAINING \
--ndepths="48,32,8"  \
--depth_inter_r="4,1,0.5" \
--dlossw="1.0,1.0,1.0" \
--loadckpt=$CKPT \
--eval_freq=1 \
--wd=0.0001 \
--nviews=4 \
--batch_size=$BATCH_SIZE \
--lr=0.0001 \
--lrepochs="6,12,14:2" \
--epochs=10 \
--trainlist=lists/plant_models/train.txt \
--testlist=lists/plant_models/test.txt \
--numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
