# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil

from src.dataset import AudioDenoisingDataset
from src.model import BudSAM
from src.metrics import mean_iou, dice_score, F1_score

import torch.multiprocessing as mp

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# Import necessary modules from the PIL library
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_SIZE = 1024
BATCH_SIZE = 8
IMG_NUM = 16

# set up parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--tr_path",
    type=str,
    default="/home/zcx/link/BirdDenoising",
    help="path to training files; four subfolders: images, masks, raw_audios, denoised_audios",
)
parser.add_argument("-task_name", type=str, default="BudSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-num_workers", type=int, default=8) #multi
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.0004, help="weight decay (default: 0.01) 0.0004"
)
parser.add_argument(
    "-lr", type=float, default=0.00005, metavar="LR", help="learning rate (absolute lr) 0.0001 0.00005"
)
parser.add_argument(
    "--resume", type=str, default="best_model/budsam_model_best.pth", help="Resuming training from checkpoint"
)

## Distributed training args
parser.add_argument("--world_size", type=int, help="world size")
parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
parser.add_argument(
    "--bucket_cap_mb",
    type=int,
    default=25,
    help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)",
)
parser.add_argument(
    "--grad_acc_steps",
    type=int,
    default=1,
    help="Gradient accumulation steps before syncing gradients for backprop",
)
parser.add_argument("--init_method", type=str, default="env://")

args = parser.parse_args()


# set up model for training

run_id = datetime.now().strftime("%Y%m%d-%H%M")
fig_save_path = join(args.work_dir, args.task_name + "-" + run_id)
os.makedirs(fig_save_path, exist_ok=True)

model_save_path = join(args.work_dir, args.task_name + "-" + run_id)


@torch.no_grad()
def budsam_inference(budsam_model, image, box_1024):
    image_embedding = budsam_model.module.image_encoder(image) 
    box_torch = torch.as_tensor(box_1024, dtype=torch.float).cuda()
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = budsam_model.module.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = budsam_model.module.mask_decoder(
        image_embeddings=image_embedding, 
        image_pe=budsam_model.module.prompt_encoder.get_dense_pe(), 
        sparse_prompt_embeddings=sparse_embeddings, 
        dense_prompt_embeddings=dense_embeddings, 
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits) 
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(image.shape[2], image.shape[3]),
        mode="bilinear",
        align_corners=False,
    ) 

    low_res_pred = low_res_pred.squeeze().cpu().numpy() 
    budsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    budsam_seg = torch.from_numpy(budsam_seg)
    return budsam_seg


def main():
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )
    
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    budsam_model = BudSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system("nvidia-smi")
    
    budsam_model = nn.parallel.DistributedDataParallel(
        budsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("After DDP initialization:")
        os.system("nvidia-smi")

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in budsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in budsam_model.parameters() if p.requires_grad),
    )  

    img_mask_encdec_params = list(budsam_model.module.image_encoder.parameters()) + list(
        budsam_model.module.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    ) 
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # %% train
    num_epochs = args.num_epochs
    # iter_num = 0
    losses = []
    best_loss = 1e10
    # 加入vitvs的评价指标
    train_losses = []
    val_losses = []
    mean_ious = []
    mean_dices = []
    mean_f1 = []
    best_iou = 0.0
    #best_model_wts = copy.deepcopy(budsam_model.state_dict())  # 有什么用？
    test_losses = []
    test_ious = []
    test_dices = []
    test_f1s = []

    train_dataset = AudioDenoisingDataset(join(args.tr_path, "train"))
    valid_dataset = AudioDenoisingDataset(join(args.tr_path, "valid"))
    test_dataset = AudioDenoisingDataset(join(args.tr_path, "test"))

    # distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=True,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    print("Number of validing samples: ", len(valid_dataset))
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler,
    )
    print("Number of testing samples: ", len(test_dataset))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    start_epoch = 0
    if args.resume is not None:
        torch.distributed.barrier()
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            
            budsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()


    for epoch in range(start_epoch, num_epochs):
        budsam_model.train()
        epoch_loss = 0
        iter_num = 0

        train_dataloader.sampler.set_epoch(epoch)

        train_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}] Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for step, (image, gt2D, boxes, _) in enumerate(train_iterator):
        # for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()
            
            budsam_pred = budsam_model(image, boxes_np)
            loss = seg_loss(budsam_pred, gt2D) + ce_loss(budsam_pred, gt2D.float())

            if args.grad_acc_steps > 1:
                loss = (
                    loss / args.grad_acc_steps
                ) 
                if (step + 1) % args.grad_acc_steps == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with budsam_model.no_sync():
                        loss.backward() 
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1
            train_iterator.set_postfix(loss=loss.item())

        train_loss = epoch_loss / iter_num
        train_losses.append(train_loss)
        
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
            1
        ] / (1024**3)
        print("\n")
        print(f"[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb")
        print(f"[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb")
        print(
            f"[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb"
        )
        print("\n")

        epoch_loss /= step
        losses.append(epoch_loss)

        print(
            f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        print(
            f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {train_loss}'
        )
        

        torch.distributed.barrier()
        if is_main_host:
            checkpoint = {
                "model": budsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "budsam_model_latest_rank0.pth"))
        else:
            checkpoint = {
                "model": budsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, f"budsam_model_latest_rank{rank}.pth"))

        torch.distributed.barrier()
        #torch.save(checkpoint, join(model_save_path, "budsam_model_{}.pth".format(epoch)))

        # ## save the best model
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     checkpoint = {
        #         "model": budsam_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch,
        #     }
        #     torch.save(checkpoint, join(model_save_path, "budsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(fig_save_path, args.task_name + f"_train_loss_rank{rank}.png"))
        plt.close()

    #  Evaluation loop for each epoch
    budsam_model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_f1 = 0
    iter_num = 0
    with torch.no_grad():
        valid_iterator = tqdm(valid_dataloader, desc="Validation", unit="batch")
        for step, (image, gt2D, boxes, _) in enumerate(valid_iterator):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()
            budsam_pred = budsam_model(image, boxes_np)
            loss = seg_loss(budsam_pred, gt2D) + ce_loss(budsam_pred, gt2D.float())
            total_loss += loss.item()
            preds = budsam_inference(budsam_model, image, boxes)

            # Compute IoU and Dice Score
            iou = mean_iou(preds, gt2D, 2)
            dice = dice_score(preds, gt2D, 2)
            f1 = F1_score(preds, gt2D, 2)
            total_iou += iou
            total_dice += dice
            total_f1 += f1
            iter_num += 1
            valid_iterator.set_postfix(loss=loss.item(), mean_iou=iou, dice_score=dice, f1_score=f1)

    valid_epoch_loss = total_loss / iter_num
    valid_epoch_iou = total_iou / iter_num
    valid_epoch_dice = total_dice / iter_num
    valid_epoch_f1 = total_f1 / iter_num
    
    val_losses.append(valid_epoch_loss)
    mean_ious.append(valid_epoch_iou)
    mean_dices.append(valid_epoch_dice)
    mean_f1.append(valid_epoch_f1)

    plt.plot(val_losses)
    plt.title("Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(fig_save_path, args.task_name + f"_val_loss_rank{rank}.png"))
    plt.close()

    plt.plot(range(len(mean_f1)), mean_f1, "brown", label="val F1 Score")
    plt.plot(range(len(mean_dices)), mean_dices, "g", label="val dice")
    plt.plot(range(len(mean_ious)), mean_ious, "orange", label="val iou")
    plt.title("Val Metric")
    plt.xlabel("Epoch")
    plt.ylabel("value(%)")
    plt.legend()
    plt.savefig(join(fig_save_path, args.task_name + f"_val_metric_rank{rank}.png"))
    plt.close()


    print(
        f"Rank{rank} Validation => Mean Loss: {valid_epoch_loss:.4f} | Mean IoU: {valid_epoch_iou:.4f} | Mean Dice: {valid_epoch_dice:.4f} | Mean F1 Score: {valid_epoch_f1:.4f}")

    print(f"Rank{rank} mean_ious: {mean_ious}")
    print(f"Rank{rank} mean_dices: {mean_dices}")
    print(f"Rank{rank} mean_f1: {mean_f1}")

    # # 确保所有进程同步
    torch.distributed.barrier()
    if is_main_host:
        checkpoint = {
            "model": budsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if valid_epoch_iou > best_iou:
            print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
            best_iou = valid_epoch_iou
            torch.save(checkpoint, join(model_save_path, f"budsam_model_best_rank{rank}.pth"))
    else:
        checkpoint = {
            "model": budsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if valid_epoch_iou > best_iou:
            print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
            best_iou = valid_epoch_iou
            torch.save(checkpoint, join(model_save_path, f"budsam_model_best_rank{rank}.pth"))

    # 确保所有进程同步
    torch.distributed.barrier()
    # """
    # Evaluation loop for each epoch
    budsam_model.eval()
    test_total_loss = 0
    test_total_iou = 0
    test_total_dice = 0
    test_total_f1 = 0
    iter_num = 0
    with torch.no_grad():
        test_iterator = tqdm(test_dataloader, desc="Test", unit="batch")
        for step, (image, gt2D, boxes, _) in enumerate(test_iterator):
            # optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()
            budsam_pred = budsam_model(image, boxes_np)
            loss = seg_loss(budsam_pred, gt2D) + ce_loss(budsam_pred, gt2D.float())
            test_total_loss += loss.item()
            preds = budsam_inference(budsam_model, image, boxes)

            # Compute IoU and Dice Score
            iou = mean_iou(preds, gt2D, 2)
            dice = dice_score(preds, gt2D, 2)
            f1 = F1_score(preds, gt2D, 2)
            test_total_iou += iou
            test_total_dice += dice
            test_total_f1 += f1
            iter_num += 1
            test_iterator.set_postfix(loss=loss.item(), mean_iou=iou, dice_score=dice, f1_score=f1)

    test_epoch_loss = test_total_loss / iter_num
    test_epoch_iou = test_total_iou / iter_num
    test_epoch_dice = test_total_dice / iter_num
    test_epoch_f1 = test_total_f1 / iter_num

    test_losses.append(test_epoch_loss)
    test_ious.append(test_epoch_iou)
    test_dices.append(test_epoch_dice)
    test_f1s.append(test_epoch_f1)

    print(
        f"Test => Mean Loss: {test_epoch_loss:.4f} | Mean IoU: {test_epoch_iou:.4f} | Mean Dice: {test_epoch_dice:.4f} | Mean F1 Score: {test_epoch_f1:.4f}")
    print(f"Rank{rank} mean_ious: {test_ious}")
    print(f"Rank{rank} mean_dices: {test_dices}")
    print(f"Rank{rank} mean_f1: {test_f1s}")


    plt.figure(figsize=(10, 7))

    # Plotting training and validation metrics
    plt.plot(range(len(train_losses)), train_losses, "b", label="train loss")
    plt.plot(range(len(val_losses)), val_losses, "r", label="val loss")
    plt.plot(range(len(mean_f1)), mean_f1, "brown", label="val F1 Score")
    plt.plot(range(len(mean_dices)), mean_dices, "g", label="val dice")
    plt.plot(range(len(mean_ious)), mean_ious, "orange", label="val iou")

    plt.legend() 
    plt.xlabel("epoch")
    
    plt.savefig(join(fig_save_path, args.task_name + "_budsam_model_best_training.png"))


if __name__ == "__main__":
    main()

