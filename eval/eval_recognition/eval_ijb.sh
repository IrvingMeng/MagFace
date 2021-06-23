CKPT=$1
FEAT_SUFFIX=$2
NL=$3

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/IJBB/meta/img.list \
                    --feat_list ${FEAT_PATH}/ijbb_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/IJBC/meta/img.list \
                    --feat_list ${FEAT_PATH}/ijbc_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}                    

echo evaluate ijbb
python3 eval_ijb.py \
        --feat_list ${FEAT_PATH}/ijbb_${FEAT_SUFFIX}.list \
        --base_dir data/IJBB/ \
        --type b \
        --embedding_size 512

echo evaluate ijbc
python3 eval_ijb.py \
        --feat_list ${FEAT_PATH}/ijbc_${FEAT_SUFFIX}.list \
        --base_dir data/IJBC/ \
        --type c \
        --embedding_size 512



