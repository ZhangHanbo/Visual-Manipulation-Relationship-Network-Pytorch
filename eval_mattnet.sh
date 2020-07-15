

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
NET="res101"
ID="rcnn_cmr_with_st"
CHECKPOINT=18301
CHECKEPOCH=13

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./eval_mattnet_easy.py \
    --ref_imdb_name ${IMDB} \
    --net_name ${NET} \
    --ref_dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --id ${ID} \
    --dataset vmrdcompv1 \
    --frame faster_rcnn \
    --net res101 \
    --cuda \
    --checkpoint ${CHECKPOINT} \
    --checkepoch ${CHECKEPOCH}
