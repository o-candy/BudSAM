# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
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
import random
from datetime import datetime
import shutil
import glob
import copy

from metrics import mean_iou, dice_score, F1_score
from torchvision.transforms.functional import resize

import torch.multiprocessing as mp

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5,6,7"

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

'''
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# %% sanity test of dataset class
tr_dataset = NpyDataset("/home/zcx/DataSet/npy/CT_Abd")
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break
'''

# %% set up audio dataset and dataloader
# Import necessary modules from the PIL library
from PIL import Image, ImageFile
from torchvision import transforms as TF
from torchvision.transforms.functional import to_tensor, resize

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set constant values for image size and batch size
IMG_SIZE = 1024 #256 1024(IoU 47)
BATCH_SIZE = 8
IMG_NUM = 16

nolabel_data = ["XC175650_left.png", "XC462632_left.png", "XC462680_left.png", "XC463223_left.png", "XC463661.png", "XC469469_left.png", "XC490972_left.png", "XC496206_left.png"]
default_xy = np.array([100, 900])

class AudioDenoisingDataset(Dataset):
    """Custom dataset for audio denoising task.

    Args:
        images_dir (str): Directory containing input images.
        masks_dir (str): Directory containing corresponding masks.
        transform (callable, optional): Optional transform to be applied on the input images.
    """

    # def __init__(self, images_dir, masks_dir, transform=None):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.images_dir = join(data_root, "Images")
        self.masks_dir = join(data_root, "Masks")
        # self.transform = transform
        self.images = [img for img in os.listdir(self.images_dir) if img.endswith('.png')]
        
        if "train" in self.data_root:
          for item in nolabel_data:
            if item in self.images:
              del self.images[self.images.index(item)]
        
        #self.images = self.images[:IMG_NUM]
        self.images = self.images
        #print(f"images list: {self.images}")
        self.masks = [mask.replace('.png', '.png') for mask in self.images]
        # self.masks = [mask for mask in os.listdir(self.masks_dir) if mask.endswith('.png')]
        self.bbox_shift = bbox_shift
        # self.transform = transform
        if nolabel_data in self.images:
          print("nolabel_data in self.images")
        if nolabel_data in self.masks:
          print("nolabel_data in self.masks")
        print(f"number of images: {len(self.images)}")
        print(f"number of masks: {len(self.masks)}")

    def __len__(self):
        """Get the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input image and its corresponding mask.
        """
        # image_path = os.path.join(self.images_dir, self.images[idx])
        # mask_path = os.path.join(self.masks_dir, self.masks[idx])
        # image = Image.open(image_path).convert("RGB")
        # mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        # # Convert mask to binary format with 0 and 1 values
        # mask = np.array(mask)
        # mask = (mask > 0).astype(np.uint8)

        # # Convert to PIL Image for consistency in transforms
        # mask = Image.fromarray(mask)

        # if self.transform:
        #     image = self.transform(image)

        # # Resize the mask to the desired image size
        # mask = resize(mask, size=[IMG_SIZE, IMG_SIZE], interpolation=Image.Resampling.NEAREST)
        # mask = TF.functional.to_tensor(mask)
        # mask = (mask > 0).long()  # Threshold back to binary and convert to LongTensor

        # return image, mask
        
        img_name = self.images[idx]
        
        # 假设输入为256
        img_ori = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask_ori = Image.open(os.path.join(self.masks_dir, img_name)).convert('L')
        img_256 = resize(img_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
        mask_256 = resize(mask_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
        #img_1024 = np.array(Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")) #(H,W,3)
        #mask = np.array(Image.open(os.path.join(self.masks_dir, img_name.replace('.png', '.png'))).convert('L'))  # Convert mask to grayscale
        img_1024 = np.array(img_256)
        mask = np.array(mask_256)
        resize_img_skimg = transform.resize(
                img_1024,
                (IMG_SIZE, IMG_SIZE),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
        resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
        resize_mask_skimg = transform.resize(
                mask,
                (IMG_SIZE, IMG_SIZE),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        resize_mask_skimg = np.uint8(resize_mask_skimg)
        resize_mask_skimg_01 = (resize_mask_skimg - resize_mask_skimg.min()) / np.clip(
                resize_mask_skimg.max() - resize_mask_skimg.min(), a_min=1e-8, a_max=None
            )

        assert resize_img_skimg_01.shape[:2] == resize_mask_skimg.shape

        # img_1024 = resize_img_skimg_01.transpose(2, 0, 1)
        img_1024 = resize_img_skimg_01
        mask = resize_mask_skimg_01

        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        
        mask = (mask > 0).astype(np.uint8)
        #print("np.max(mask): {}, np.min(mask): {}".format(np.max(mask), np.min(mask)))
        if not (np.max(mask) == 1 and np.min(mask) == 0.0):
            print("img_name: {}".format(img_name))
            
        #assert np.max(mask) == 1 and np.min(mask) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(mask > 0)
        if y_indices.size == 0:
            y_indices = default_xy
        if x_indices.size == 0:
            x_indices = default_xy
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )



# # Define the appropriate transformations
# transform = TF.Compose([
#     TF.Resize((IMG_SIZE, IMG_SIZE)),
#     # Uncomment the following lines if normalization is needed
#     # TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     TF.ToTensor()
# ])

# Define the dataset paths
# # Train part
# train_images_dir = 'dataset/Train/Images'
# train_masks_dir = 'dataset/Train/Masks'

# # Validation part
# valid_images_dir = 'dataset/Valid/Images'
# valid_masks_dir = 'dataset/Valid/Masks'

# # Test part
# test_images_dir = 'dataset/Test/Images'
# test_masks_dir = 'dataset/Test/Masks'
'''
train_data_root = "/data/zcx_data/BirdDenoising/train"
valid_data_root = "/data/zcx_data/BirdDenoising/valid"
test_data_root = "/data/zcx_data/BirdDenoising/test"

# Create the datasets
train_dataset = AudioDenoisingDataset(data_root=train_data_root)
valid_dataset = AudioDenoisingDataset(data_root=valid_data_root)
test_dataset = AudioDenoisingDataset(data_root=test_data_root)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for step, (image, gt, bboxes, names_temp) in enumerate(train_loader):
    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck_BD.png", bbox_inches="tight", dpi=300)
    plt.close()
    break
'''
# %% set up parser
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-i",
#     "--tr_npy_path",
#     type=str,
#     default="data/npy/CT_Abd",
#     help="path to training npy files; two subfolders: gts and imgs",
# )
parser.add_argument(
    "-i",
    "--tr_path",
    type=str,
    default="/home/zcx/link/zcx/BirdDenoising/",
    help="path to training files; four subfolders: images, masks, raw_audios, denoised_audios",
)
parser.add_argument("-task_name", type=str, default="AudSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
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
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint /data/zcx_data/models/AudSAM-ViT-B-2GPUs-20240824-0026/audsam_model_latest.pth"
)
# parser.add_argument("--device", type=str, default="cuda:0")
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

# node_rank = int(args.node_rank)
# ngpus_per_node = torch.cuda.device_count()

# rank = node_rank * ngpus_per_node + gpu

# is_main_host = rank == 0

# if is_main_host:
if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
# model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
fig_save_path = join(args.work_dir, args.task_name + "-" + run_id)
os.makedirs(fig_save_path, exist_ok=True)

# models_path = "/data/zcx_data/models/"
# os.makedirs(models_path, exist_ok=True)
# model_save_path = join(models_path, args.task_name + "-" + run_id)
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
# device = torch.device(args.device)
# %% set up model


class AudSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

@torch.no_grad()
def audsam_inference(audsam_model, image, box_1024):
    image_embedding = audsam_model.module.image_encoder(image)  # (B, 256, 64, 64)
    box_torch = torch.as_tensor(box_1024, dtype=torch.float).cuda()
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = audsam_model.module.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = audsam_model.module.mask_decoder(
        image_embeddings=image_embedding,  # (B, 256, 64, 64)
        image_pe=audsam_model.module.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    #print(f"low_res_pred sigmoid: {low_res_pred.shape}, {low_res_pred}")
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(image.shape[2], image.shape[3]),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    #low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    #print(f"low_res_pred: {low_res_pred.shape}, {low_res_pred}")
    audsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    #print(f"audsam_seg: {audsam_seg.shape}, {audsam_seg}")
    audsam_seg = torch.from_numpy(audsam_seg)
    return audsam_seg


def main():
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    # os.makedirs(model_save_path, exist_ok=True)
    # shutil.copyfile(
    #     __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    # )
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
    # device = torch.device("cuda:{}".format(gpu))
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )
    
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    audsam_model = AudSAM(
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
    
    audsam_model = nn.parallel.DistributedDataParallel(
        audsam_model,
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
    
    '''
    audsam_model = AudSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    )
    audsam_model = nn.DataParallel(audsam_model)
    audsam_model.to(device)
    '''
    # audsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in audsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in audsam_model.parameters() if p.requires_grad),
    )  # 93729252
    '''
    img_mask_encdec_params = list(audsam_model.image_encoder.parameters()) + list(
        audsam_model.mask_decoder.parameters()
    )
    '''
    img_mask_encdec_params = list(audsam_model.module.image_encoder.parameters()) + list(
        audsam_model.module.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
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
    #best_model_wts = copy.deepcopy(audsam_model.state_dict())  # 有什么用？
    # test_losses = []
    # test_ious = []
    # test_dices = []
    # test_f1s = []

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
            # ## Map model to be loaded to specified single GPU
            # checkpoint = torch.load(args.resume, map_location=device)
            # start_epoch = checkpoint["epoch"] + 1
            # audsam_model.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            audsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    for epoch in range(start_epoch, num_epochs):
        audsam_model.train()
        epoch_loss = 0
        iter_num = 0

        train_dataloader.sampler.set_epoch(epoch)

        train_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}] Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for step, (image, gt2D, boxes, _) in enumerate(train_iterator):
        # for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    audsam_pred = audsam_model(image, boxes_np)
                    loss = seg_loss(audsam_pred, gt2D) + ce_loss(
                        audsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                audsam_pred = audsam_model(image, boxes_np)
                loss = seg_loss(audsam_pred, gt2D) + ce_loss(audsam_pred, gt2D.float())
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # Gradient accumulation
                if args.grad_acc_steps > 1:
                    loss = (
                        loss / args.grad_acc_steps
                    )  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with audsam_model.no_sync():
                            loss.backward()  ## calculate the gradient only
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1
            train_iterator.set_postfix(loss=loss.item())

        train_loss = epoch_loss / iter_num
        train_losses.append(train_loss)
        
        # Check CUDA memory usage
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
        '''
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
            wandb.log({"train_loss": train_loss})
        '''
        print(
            f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        print(
            f'Rank{rank} Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {train_loss}'
        )
        
        # 确保所有进程同步
        torch.distributed.barrier()
        if is_main_host:
            ## save the latest model
            checkpoint = {
                "model": audsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "audsam_model_latest_rank0.pth"))
        else:
            checkpoint = {
                "model": audsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, f"audsam_model_latest_rank{rank}.pth"))
        # 确保所有进程同步
        torch.distributed.barrier()
        #torch.save(checkpoint, join(model_save_path, "audsam_model_{}.pth".format(epoch)))

        # ## save the best model
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     checkpoint = {
        #         "model": audsam_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch,
        #     }
        #     torch.save(checkpoint, join(model_save_path, "audsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(fig_save_path, args.task_name + f"_train_loss_rank{rank}.png"))
        plt.close()

        # %% Evaluation loop for each epoch
        audsam_model.eval()
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
                if args.use_amp:
                    ## AMP
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        audsam_pred = audsam_model(image, boxes_np)
                        loss = seg_loss(audsam_pred, gt2D) + ce_loss(
                            audsam_pred, gt2D.float()
                        )
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    # optimizer.zero_grad()
                else:
                    audsam_pred = audsam_model(image, boxes_np)
                    loss = seg_loss(audsam_pred, gt2D) + ce_loss(audsam_pred, gt2D.float())
                    # loss.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()

                total_loss += loss.item()

                # Get the logits from the model and apply argmax to get the predictions
                # outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)
                '''
                preds = torch.argmax(audsam_pred, dim=1)
                preds = torch.unsqueeze(preds, dim=1)

                preds = preds.view(-1)
                gt2D = gt2D.view(-1)
                '''
                preds = audsam_inference(audsam_model, image, boxes)

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

        
        
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
            wandb.log({"train_loss": train_loss})
            wandb.log({"valid_epoch_loss": valid_epoch_loss})
            wandb.log({"valid_epoch_iou": valid_epoch_iou})
            wandb.log({"valid_epoch_dice": valid_epoch_dice})
            wandb.log({"valid_epoch_f1": valid_epoch_f1})
        
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

        # 确保所有进程同步
        torch.distributed.barrier()
        if is_main_host:
            checkpoint = {
                "model": audsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            # Check for improvement and save the best model weights based on IoU
            if valid_epoch_iou > best_iou:
                print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
                best_iou = valid_epoch_iou
                #best_model_wts = copy.deepcopy(audsam_model.state_dict())
                torch.save(checkpoint, join(model_save_path, f"audsam_model_best_rank{rank}.pth"))
        else:
            checkpoint = {
                "model": audsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            # Check for improvement and save the best model weights based on IoU
            if valid_epoch_iou > best_iou:
                print(f"Rank{rank} Validation IoU improved from {best_iou:.4f} to {valid_epoch_iou:.4f}")
                best_iou = valid_epoch_iou
                #best_model_wts = copy.deepcopy(audsam_model.state_dict())
                torch.save(checkpoint, join(model_save_path, f"audsam_model_best_rank{rank}.pth"))

        # 确保所有进程同步
        torch.distributed.barrier()
        """
        # %% Evaluation loop for each epoch
        audsam_model.eval()
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
                if args.use_amp:
                    ## AMP
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        audsam_pred = audsam_model(image, boxes_np)
                        loss = seg_loss(audsam_pred, gt2D) + ce_loss(
                            audsam_pred, gt2D.float()
                        )
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    # optimizer.zero_grad()
                else:
                    audsam_pred = audsam_model(image, boxes_np)
                    loss = seg_loss(audsam_pred, gt2D) + ce_loss(audsam_pred, gt2D.float())
                    # loss.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()

                test_total_loss += loss.item()

                # Get the logits from the model and apply argmax to get the predictions
                # outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)
                '''
                preds = torch.argmax(audsam_pred, dim=1)
                preds = torch.unsqueeze(preds, dim=1)

                preds = preds.view(-1)
                gt2D = gt2D.view(-1)
                '''
                preds = audsam_inference(audsam_model, image, boxes)

                # Compute IoU and Dice Score
                iou = mean_iou(preds, gt2D, 2)
                dice = dice_score(preds, gt2D, 2)
                f1 = F1_score(preds, gt2D, 2)
                total_iou += iou
                total_dice += dice
                total_f1 += f1

                iter_num += 1

                test_iterator.set_postfix(loss=loss.item(), mean_iou=iou, dice_score=dice, f1_score=f1)

        test_epoch_loss = test_total_loss / iter_num
        test_epoch_iou = test_total_iou / iter_num
        test_epoch_dice = test_total_dice / iter_num
        test_epoch_f1 = test_total_f1 / iter_num
        
        if args.use_wandb:
            wandb.log({"test_epoch_loss": test_epoch_loss})
            wandb.log({"test_epoch_iou": test_epoch_iou})
            wandb.log({"test_epoch_dice": test_epoch_dice})
            wandb.log({"test_epoch_f1": test_epoch_f1})
        
        test_losses.append(test_epoch_loss)
        test_ious.append(test_epoch_iou)
        test_dices.append(test_epoch_dice)
        test_f1s.append(test_epoch_f1)

        print(
            f"Test => Mean Loss: {test_epoch_loss:.4f} | Mean IoU: {test_epoch_iou:.4f} | Mean Dice: {test_epoch_dice:.4f} | Mean F1 Score: {test_epoch_f1:.4f}")
        """

        # After all epochs, load the best model weights - optional
    #audsam_model.load_state_dict(torch.load(join(model_save_path, "audsam_model_best.pth")))
    #print("Loaded the best model weights!")

    plt.figure(figsize=(10, 7))

    # Plotting training and validation metrics
    plt.plot(range(len(train_losses)), train_losses, "b", label="train loss")
    plt.plot(range(len(val_losses)), val_losses, "r", label="val loss")
    plt.plot(range(len(mean_f1)), mean_f1, "brown", label="val F1 Score")
    plt.plot(range(len(mean_dices)), mean_dices, "g", label="val dice")
    plt.plot(range(len(mean_ious)), mean_ious, "orange", label="val iou")

    plt.legend()  # Adding legend
    plt.xlabel("epoch")  # Labeling x-axis with 'epoch'
    
    plt.savefig(join(fig_save_path, args.task_name + "_audsam_model_best_training.png"))


if __name__ == "__main__":
    main()
    '''
    python audsam_train_multi_gpu_v3.py \
        -task_name AudSAM-ViT-B-2GPUs-lr5e-5 \
        -work_dir ./work_dir \
        -batch_size 4 \
        -num_workers 2 \
        --world_size 2 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344
        
    nohup python audsam_train_multi_gpu_v3.py \
        -task_name AudSAM-ViT-B-6GPUs-lr5e-5-4mask-2nd-auto1 \
        -work_dir ./work_dir \
        -batch_size 2 \
        -num_workers 6 \
        --world_size 6 \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 \
        --init_method tcp://localhost:12344 > ./logs/auto1_train_multiv3_904_135_4mask.log 2>&1 &
        lr5e-5
        1e-3 0.01 4gpu
    '''

