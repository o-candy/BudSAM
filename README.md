# Segment Anything for Visual Bird Sound Denoising

Implementation of "Segment Anything for Visual Bird Sound Denoising".

## Abstract
Current audio denoising methods perform well with synthetic noise but struggle with complex natural noise, especially for bird sounds, which contain natural environmental sounds such as wind and rain, making it challenging to extract clean bird sounds. This issue becomes more pronounced with short and faint bird sounds, where existing methods are less effective. In this paper, we introduce BudSAM, a novel audio denoising model that incorporates the Segment Anything Model (SAM), originally designed for image segmentation task, into the field of visual bird sound denoising. By treating audio denoising as a segmentation task, BudSAM leverages SAM’s powerful segmentation capabilities and we incorporates BCE and Dice losses to enhance the model’s ability to segment weak signals, effectively isolating the clean bird sounds that are often masked by background noise. Our method is evaluated on the BirdSoundsDenoising dataset, achieving a 4.0\% improvement in IoU and a 0.77 dB increase in SDR compared to state-of-the-art methods. To the best knowledge of the authors, BudSAM marks the first attempt which employs SAM in audio denoising task, offering a promising direction for future research and real-world bird sound processing tasks.

## Installation
    pip install -r requirements.txt

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