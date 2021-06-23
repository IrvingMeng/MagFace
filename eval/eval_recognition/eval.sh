CKPT=$1
FEAT_SUFFIX=$2
NL=$3

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw/img.list \
                    --feat_list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/cfp/img.list \
                    --feat_list ${FEAT_PATH}/cfp_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb/img.list \
                    --feat_list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

echo evaluate lfw
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
		--pair-list data/lfw/pair.list \


echo evaluate cfp
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/cfp_${FEAT_SUFFIX}.list \
		--pair-list data/cfp/pair.list \

echo evaluate agedb
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
		--pair-list data/agedb/pair.list \