# Segment Anything for Visual Bird Sound Denoising

Implementation of "Segment Anything for Visual Bird Sound Denoising".


## Installation
    pip install -r requirements.txt

## Model
We use the [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to initialize the BudSAM model parameters.

We provide access to our trained [BudSAM model](https://pan.baidu.com/s/159-UM-_WpqWa87nTaWoaUA?pwd=g6np).

## Run
    nohup python budsam_train_multi.py \
        -task_name BudSAM-test-5GPUs \
        -work_dir ./work_dir \
        -batch_size 2 \
        -num_workers 5 \
        --world_size 5 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344 > ./logs/run.log 2>&1 &

## Acknowledgements
- We thank the BirdSoundsDenoising dataset contributors for providing the dataset to the community.
- We thank for the source code of [Segment Anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We thank for the source code of [Segment anything in medical images](https://github.com/bowang-lab/MedSAM) publicly available.