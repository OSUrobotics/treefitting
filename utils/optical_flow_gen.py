import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from FileNames import FileNames
import torchvision.transforms.functional as FV
import torch.nn.functional as F


class ImagePairDataset(Dataset):
    def __init__(self, filenames: FileNames, skip=1):
        self.file_instance = filenames
        self.path_idxs = [f for f in filenames.loop_images()]
        filenames.check_names()
        self.skip = skip

        # resize to multiples of 8
        height, width = cv2.imread(filenames.get_image_name(self.path_idxs[0])).shape[
            :2
        ]
        height = min(320, height - height % 8)
        width = min(576, width - width % 8)
        self.size = [height, width]

    def __len__(self):
        return (len(self.path_idxs) - 1) // self.skip

    def __getitem__(self, idx):
        img1_idx = idx * self.skip
        img2_idx = img1_idx + 1
        img1_path = self.file_instance.get_image_name(self.path_idxs[img1_idx])
        img2_path = self.file_instance.get_image_name(self.path_idxs[img2_idx])

        img1 = torch.from_numpy(cv2.imread(img1_path)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(cv2.imread(img2_path)).permute(2, 0, 1).float()
        # img_name and remove the extension
        flow_name = self.file_instance.get_edge_name(
            self.path_idxs[img1_idx], b_optical_flow=True
        )

        return img1, img2, flow_name


def preprocess_raft(img1_batch, img2_batch, transforms, size):
    img1_batch = FV.resize(img1_batch / 255.0, size=size, antialias=False)
    img2_batch = FV.resize(img2_batch / 255.0, size=size, antialias=False)
    return transforms(img1_batch, img2_batch)


def preprocess_unimatch(image1, image2, inference_size):
    image1 = F.interpolate(
        image1, size=inference_size, mode="bilinear", align_corners=True
    )
    image2 = F.interpolate(
        image2, size=inference_size, mode="bilinear", align_corners=True
    )
    return image1, image2


def fetch_image_pairs(filenames, transforms, skip=1, batch_size=4, num_workers=4):
    dataset = ImagePairDataset(filenames, skip)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    img1_l, img2_l, filenames = [], [], []
    for img1_batch, img2_batch, f in dataloader:
        img1_batch, img2_batch = preprocess_raft(
            img1_batch, img2_batch, transforms, dataset.size
        )
        img1_l.append(img1_batch)
        img2_l.append(img2_batch)
        filenames.extend(f)

    img1_batch = torch.cat(img1_l, dim=0)
    img2_batch = torch.cat(img2_l, dim=0)
    return img1_batch, img2_batch, filenames


@torch.no_grad()
def load_and_infer_raft(filenames, batch_size=1, skip=1):
    """_summary_

    :param filenames: FileNames object
    :param batch_size: inference batch size, defaults to 1
    :param skip: how many files to skip if on high freq cameras, defaults to 1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    from torchvision.utils import flow_to_image

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval()

    dataset = ImagePairDataset(filenames, skip)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for img1_batch, img2_batch, fs in dataloader:
        img1_batch, img2_batch = preprocess_raft(
            img1_batch, img2_batch, transforms, dataset.size
        )
        predicted_flows = model(img1_batch.to(device), img2_batch.to(device))
        predicted_flows = list_of_flows[-1]

        flow_imgs = flow_to_image(predicted_flows).permute(0, 2, 3, 1)
        for flow_img, filename in zip(flow_imgs, fs):
            print(f"{filename} shape = {flow_img.shape}")
            cv2.imwrite(filename, flow_img.cpu().numpy())

        del img1_batch, img2_batch, list_of_flows, predicted_flows, flow_imgs
        torch.cuda.empty_cache()


# borrowed from evaluate_flow/inference_flow
@torch.no_grad()
def load_and_infer_unimatch(filenames, checkpoint_path, batch_size=1, skip=1):
    """unimatch inference
    :param filenames: FileNames object
    :param checkpoint: checkpoint path
    :param batch_size: inference batch size, defaults to 1
    :param skip: how many files to skip if on high freq cameras, defaults to 1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    import sys
    import os

    sys.path.append(os.path.join(os.path.expanduser("~"), "models", "unimatch"))
    from unimatch.unimatch import UniMatch
    from utils.flow_viz import save_vis_flow_tofile, flow_to_image

    model = UniMatch(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=False,
        task="flow",
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    dataset = ImagePairDataset(filenames, skip)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for img1_batch, img2_batch, fs in dataloader:
        img1_batch, img2_batch = preprocess_unimatch(
            img1_batch, img2_batch, dataset.size
        )
        results_dict = model(
            img1_batch.to(device),
            img2_batch.to(device),
            attn_type="swin",
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            num_reg_refine = 1,
            task = 'flow',
        )
        predicted_flows = results_dict["flow_preds"][-1]
        
        for pf, filename in zip(predicted_flows, fs):
            flow_img = flow_to_image(pf.permute(1, 2, 0).cpu().numpy())
            print(f"{filename} shape = {flow_img.shape}")
            cv2.imwrite(filename, flow_img)

        del img1_batch, img2_batch, results_dict, predicted_flows, flow_img
        torch.cuda.empty_cache()


if __name__ == "__main__":
    all_fnames = FileNames.read_filenames(
        path="/home/roosh/training_data/envy/fns/",
        fname="/home/roosh/training_data/envy/fns/envy_fnames.json",
    )
    load_and_infer_unimatch(
        all_fnames,
        "/home/roosh/models/unimatch/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth",
        1,
        1,
    )
