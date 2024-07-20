GPUID=0
INFEATURES=2
OUTFEATURES=3
HIDDENLAYERS=3
HIDDENFEATURES=256
DATADIR="data/div2k/test_data" 
IMAGEID=1
NUM_EPOCHS=5000


# Sine
# CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
#     --model_type Siren \
#     --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
#     --first_omega 30 --hidden_omega 30 \
#     --init_method sine \
#     --lr 5e-4 --num_epochs $NUM_EPOCHS \
#     --logdir logs/sine # --test


# Finer++Sine
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type Finer \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --first_omega 30 --hidden_omega 30 \
    --init_method sine \
    --lr 5e-4 --num_epochs $NUM_EPOCHS \
    --logdir logs/finer++sine --test

# Gauss.
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type Gauss \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --scale 30 \
    --init_method pytorch \
    --lr 5e-3 --num_epochs $NUM_EPOCHS \
    --logdir logs/gauss --test

# Finer++Gauss
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type GF \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --scale 10 --omega 3 \
    --init_method sine \
    --lr 1e-3 --num_epochs $NUM_EPOCHS \
    --logdir logs/finer++gauss --test

# Wavelet
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type Wire \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --scale 10 --omega_w 20 \
    --init_method pytorch \
    --lr 5e-3 --num_epochs $NUM_EPOCHS \
    --logdir logs/wavelet --test

# Finer++Wavelet
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type WF \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --scale 2 --omega_w 4 --omega 5 \
    --init_method sine \
    --lr 1e-3 --num_epochs $NUM_EPOCHS \
    --logdir logs/finer++wavelet --test


# PEMLP
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMAGEID --datadir $DATADIR \
    --model_type PEMLP \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
    --N_freqs 10 \
    --init_method pytorch \
    --lr 5e-4 --num_epochs $NUM_EPOCHS \
    --logdir logs/pemlp --test