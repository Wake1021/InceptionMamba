DATA_PATH=/path/to/ImageNet
CODE_PATH=/path/to/code/inceptionmamba # modify code path here

NUM_GPU=8
BATCH_SIZE=64

MODEL=inceptionmamba_base
DROP_PATH=0.4

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 1e-3 --warmup-epochs 20 \
-b $BATCH_SIZE \
--drop-path $DROP_PATH \
--min-lr 5e-6 \
--pin-mem \
--amp