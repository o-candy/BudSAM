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

nolabel_data = ["XC175650_left.png", "XC462632_left.png", "XC462680_left.png", "XC463223_left.png", "XC463661.png", "XC469469_left.png", "XC490972_left.png", "XC496206_left.png"]
default_xy = np.array([100, 900])

# class AudioDenoisingDataset(Dataset):
#     def __init__(self, data_root, bbox_shift=20):
#         self.data_root = data_root
#         self.images_dir = join(data_root, "Images")
#         self.masks_dir = join(data_root, "Masks")
#         self.images = [img for img in os.listdir(self.images_dir) if img.endswith('.png')]
        
#         if "train" in self.data_root:
#           for item in nolabel_data:
#             if item in self.images:
#               del self.images[self.images.index(item)]
        
#         #self.images = self.images[:IMG_NUM]
#         self.images = self.images
#         #print(f"images list: {self.images}")
#         self.masks = [mask.replace('.png', '.png') for mask in self.images]
#         # self.masks = [mask for mask in os.listdir(self.masks_dir) if mask.endswith('.png')]
#         self.bbox_shift = bbox_shift
#         # self.transform = transform
#         if nolabel_data in self.images:
#           print("nolabel_data in self.images")
#         if nolabel_data in self.masks:
#           print("nolabel_data in self.masks")
#         print(f"number of images: {len(self.images)}")
#         print(f"number of masks: {len(self.masks)}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
        
#         img_ori = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
#         mask_ori = Image.open(os.path.join(self.masks_dir, img_name)).convert('L')
#         img_256 = resize(img_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
#         mask_256 = resize(mask_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
#         img_1024 = np.array(img_256)
#         mask = np.array(mask_256)
#         resize_img_skimg = transform.resize(
#                 img_1024,
#                 (IMG_SIZE, IMG_SIZE),
#                 order=3,
#                 preserve_range=True,
#                 mode="constant",
#                 anti_aliasing=True,
#             )
#         resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
#                 resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
#             )  # normalize to [0, 1], (H, W, 3)
#         resize_mask_skimg = transform.resize(
#                 mask,
#                 (IMG_SIZE, IMG_SIZE),
#                 order=0,
#                 preserve_range=True,
#                 mode="constant",
#                 anti_aliasing=False,
#             )
#         resize_mask_skimg = np.uint8(resize_mask_skimg)
#         resize_mask_skimg_01 = (resize_mask_skimg - resize_mask_skimg.min()) / np.clip(
#                 resize_mask_skimg.max() - resize_mask_skimg.min(), a_min=1e-8, a_max=None
#             )

#         assert resize_img_skimg_01.shape[:2] == resize_mask_skimg.shape

#         img_1024 = resize_img_skimg_01
#         mask = resize_mask_skimg_01

#         img_1024 = np.transpose(img_1024, (2, 0, 1))
#         assert (
#             np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
#         ), "image should be normalized to [0, 1]"
        
#         mask = (mask > 0).astype(np.uint8)
#         if not (np.max(mask) == 1 and np.min(mask) == 0.0):
#             print("img_name: {}".format(img_name))
            
#         #assert np.max(mask) == 1 and np.min(mask) == 0.0, "ground truth should be 0, 1"
#         y_indices, x_indices = np.where(mask > 0)
#         if y_indices.size == 0:
#             y_indices = default_xy
#         if x_indices.size == 0:
#             x_indices = default_xy
        
#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)
#         # add perturbation to bounding box coordinates
#         H, W = mask.shape
#         x_min = max(0, x_min - random.randint(0, self.bbox_shift))
#         x_max = min(W, x_max + random.randint(0, self.bbox_shift))
#         y_min = max(0, y_min - random.randint(0, self.bbox_shift))
#         y_max = min(H, y_max + random.randint(0, self.bbox_shift))
#         bboxes = np.array([x_min, y_min, x_max, y_max])
        
#         return (
#             torch.tensor(img_1024).float(),
#             torch.tensor(mask[None, :, :]).long(),
#             torch.tensor(bboxes).float(),
#             img_name,
#         )

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


# class BudSAM(nn.Module):
#     def __init__(
#         self,
#         image_encoder,
#         mask_decoder,
#         prompt_encoder,
#     ):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.mask_decoder = mask_decoder
#         self.prompt_encoder = prompt_encoder
#         # freeze prompt encoder
#         for param in self.prompt_encoder.parameters():
#             param.requires_grad = False

#     def forward(self, image, box):
#         image_embedding = self.image_encoder(image) 
#         # do not compute gradients for prompt encoder
#         with torch.no_grad():
#             box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
#             if len(box_torch.shape) == 2:
#                 box_torch = box_torch[:, None, :]

#             sparse_embeddings, dense_embeddings = self.prompt_encoder(
#                 points=None,
#                 boxes=box_torch,
#                 masks=None,
#             )
#         low_res_masks, _ = self.mask_decoder(
#             image_embeddings=image_embedding, 
#             image_pe=self.prompt_encoder.get_dense_pe(), 
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings, 
#             multimask_output=False,
#         )
#         ori_res_masks = F.interpolate(
#             low_res_masks,
#             size=(image.shape[2], image.shape[3]),
#             mode="bilinear",
#             align_corners=False,
#         )
#         return ori_res_masks

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
            # ## Map model to be loaded to specified single GPU
            # checkpoint = torch.load(args.resume, map_location=device)
            # start_epoch = checkpoint["epoch"] + 1
            # budsam_model.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            
            budsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # state_dict = checkpoint["model"]
            # for key in list(state_dict.keys()):
            #     if key.startswith('module.'):
            #         state_dict[key.replace('module.', '')] = state_dict.pop(key)
            # budsam_model.load_state_dict(state_dict, strict=False)
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()


    
    # for epoch in range(start_epoch, num_epochs):
    #     budsam_model.train()
    #     epoch_loss = 0
    #     iter_num = 0

    #     train_dataloader.sampler.set_epoch(epoch)

    #     train_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}] Epoch {epoch + 1}/{num_epochs}", unit="batch")
    #     for step, (image, gt2D, boxes, _) in enumerate(train_iterator):
    #     # for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
    #         optimizer.zero_grad()
    #         boxes_np = boxes.detach().cpu().numpy()
    #         image, gt2D = image.cuda(), gt2D.cuda()
    #         if args.use_amp:
    #             ## AMP
    #             with torch.autocast(device_type="cuda", dtype=torch.float16):
    #                 budsam_pred = budsam_model(image, boxes_np)
    #                 loss = seg_loss(budsam_pred, gt2D) + ce_loss(
    #                     budsam_pred, gt2D.float()
    #                 )
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad()
    #         else:
    #             budsam_pred = budsam_model(image, boxes_np)
    #             loss = seg_loss(budsam_pred, gt2D) + ce_loss(budsam_pred, gt2D.float())
    #             # loss.backward()
    #             # optimizer.step()
    #             # optimizer.zero_grad()
    #             # Gradient accumulation
    #             if args.grad_acc_steps > 1:
    #                 loss = (
    #                     loss / args.grad_acc_steps
    #                 )  # normalize the loss because it is accumulated
    #                 if (step + 1) % args.grad_acc_steps == 0:
    #                     ## Perform gradient sync
    #                     loss.backward()
    #                     optimizer.step()
    #                     optimizer.zero_grad()
    #                 else:
    #                     ## Accumulate gradient on current node without backproping
    #                     with budsam_model.no_sync():
    #                         loss.backward()  ## calculate the gradient only
    #             else:
    #                 loss.backward()
    #                 optimizer.step()
    #                 optimizer.zero_grad()

    #         epoch_loss += loss.item()
    #         iter_num += 1
    #         train_iterator.set_postfix(loss=loss.item())

    #     train_loss = epoch_loss / iter_num
    #     train_losses.append(train_loss)
        
    #     # Check CUDA memory usage
    #     cuda_mem_info = torch.cuda.mem_get_info(gpu)
    #     free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
    #         1
    #     ] / (1024**3)
    #     print("\n")
    #     print(f"[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb")
    #     print(f"[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb")
    #     print(
    #         f"[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb"
    #     )
    #     print("\n")

    #     epoch_loss /= step
    #     losses.append(epoch_loss)
    #     '''
    #     if args.use_wandb:
    #         wandb.log({"epoch_loss": epoch_loss})
    #         wandb.log({"train_loss": train_loss})
    #     '''
    #     print(
    #         f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
    #     )
    #     print(
    #         f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {train_loss}'
    #     )
        
    #     # 确保所有进程同步
    #     torch.distributed.barrier()
    #     if is_main_host:
    #         ## save the latest model
    #         checkpoint = {
    #             "model": budsam_model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "epoch": epoch,
    #         }
    #         torch.save(checkpoint, join(model_save_path, "budsam_model_latest_rank0.pth"))
    #     else:
    #         checkpoint = {
    #             "model": budsam_model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "epoch": epoch,
    #         }
    #         torch.save(checkpoint, join(model_save_path, f"budsam_model_latest_rank{rank}.pth"))
    #     # 确保所有进程同步
    #     torch.distributed.barrier()
    #     #torch.save(checkpoint, join(model_save_path, "budsam_model_{}.pth".format(epoch)))

    #     # ## save the best model
    #     # if epoch_loss < best_loss:
    #     #     best_loss = epoch_loss
    #     #     checkpoint = {
    #     #         "model": budsam_model.state_dict(),
    #     #         "optimizer": optimizer.state_dict(),
    #     #         "epoch": epoch,
    #     #     }
    #     #     torch.save(checkpoint, join(model_save_path, "budsam_model_best.pth"))

    #     # %% plot loss
    #     plt.plot(losses)
    #     plt.title("Dice + Cross Entropy Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.savefig(join(fig_save_path, args.task_name + f"_train_loss_rank{rank}.png"))
    #     plt.close()

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
            # optimizer.zero_grad()
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
    # torch.distributed.barrier()
    # if is_main_host:
    #     checkpoint = {
    #         "model": budsam_model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epoch": epoch,
    #     }
    #     # Check for improvement and save the best model weights based on IoU
    #     if valid_epoch_iou > best_iou:
    #         print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
    #         best_iou = valid_epoch_iou
    #         #best_model_wts = copy.deepcopy(budsam_model.state_dict())
    #         torch.save(checkpoint, join(model_save_path, f"budsam_model_best_rank{rank}.pth"))
    # else:
    #     checkpoint = {
    #         "model": budsam_model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epoch": epoch,
    #     }
    #     # Check for improvement and save the best model weights based on IoU
    #     if valid_epoch_iou > best_iou:
    #         print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
    #         best_iou = valid_epoch_iou
    #         #best_model_wts = copy.deepcopy(budsam_model.state_dict())
    #         torch.save(checkpoint, join(model_save_path, f"budsam_model_best_rank{rank}.pth"))

    # # 确保所有进程同步
    # torch.distributed.barrier()
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
    
    # """

        # After all epochs, load the best model weights - optional
    #budsam_model.load_state_dict(torch.load(join(model_save_path, "budsam_model_best.pth")))
    #print("Loaded the best model weights!")

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
    '''
    python budsam_train_multi_gpu_v3.py \
        -task_name BudSAM-ViT-B-2GPUs-lr5e-5 \
        -work_dir ./work_dir \
        -batch_size 4 \
        -num_workers 2 \
        --world_size 2 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344
        
    nohup python budsam_test_metrics_multi.py \
        -task_name BudSAM-test-5auto1-2nd \
        -work_dir ./work_dir \
        -batch_size 2 \
        -num_workers 5 \
        --world_size 5 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344 > ./logs/auto1_test_multiv3_904_5_2032.log 2>&1 &
        lr5e-5
        1e-3 0.01 4gpu

        # mask decoder test
        nohup python budsam_train_multi.py \
        -task_name BudSAM-test-5GPUs \
        -work_dir ./work_dir \
        -batch_size 2 \
        -num_workers 5 \
        --world_size 5 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344 > ./logs/budsam_test_multi_1207.log 2>&1 &
    '''

