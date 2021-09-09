DATA=$1
NAME=official

echo evaluate ${DATA}
python3 eval_quality.py \
        --feat-list ../eval_recognition/features/magface_iresnet100/${DATA}_${NAME}.list \
        --pair-list ../eval_recognition/data/${DATA}/pair.list