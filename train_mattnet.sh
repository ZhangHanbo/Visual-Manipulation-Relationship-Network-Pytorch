

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
ID="rcnn_cmr_with_st_from_pretrained"
CHECKPOINT=18301
CHECKEPOCH=13

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_mattnet.py \
    --ref_imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --ref_dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_iters 30000 \
    --with_st 1 \
    --id ${ID} \
    --dataset vmrdcompv1 \
    --frame faster_rcnn \
    --net res101 \
    --cuda \
    --checkpoint ${CHECKPOINT} \
    --checkepoch ${CHECKEPOCH} \
    --jemb_drop_out 0.2 \
    --visual_drop_out 0.25 \
    --rnn_drop_out 0.25 \
    --start_from 'rcnn_cmr_with_st' \
    --learning_rate 4e-4
