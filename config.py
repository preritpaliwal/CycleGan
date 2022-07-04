from re import T
from sklearn.covariance import fast_mcd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "datasets/horse2zebra"
TEST_DIR = "datasets/horse2zebra"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_INDENTITY = 0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genz_checkpoint.pth"
CHECKPOINT_GEN_Z = "genh_checkpoint.pth"
CHECKPOINT_CRITIC_H = "criticz_checkpoint.pth"
CHECKPOINT_CRITIC_Z = "critich_checkpoint.pth"

transforms = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)