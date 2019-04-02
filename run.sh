SYNT_DIR="/srv/workplace/tjurica/data/simgan_defects/artificial"
REAL_DIR="/srv/workplace/tjurica/data/simgan_defects/real"

CUDA_VISIBLE_DEVICES=0 python3 sim-gan.py ${SYNT_DIR} ${REAL_DIR}