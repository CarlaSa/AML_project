import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser, Namespace

from network.Model import OurModel
from network.unet import Unet
from train.unet import Criterion
from datasets import TestData
from utils import BoundingBoxes, CanvasTrafoRecorder


def optimal_batch_size(data: Dataset, model: OurModel, start: int = 16):
    candidate: int = start
    working: int = 0
    too_high: int = 0
    while too_high == 0 or working < candidate:
        torch.cuda.empty_cache()
        print("trying", candidate)
        dataloader = DataLoader(data, batch_size=candidate)
        try:
            for x, _, _, _ in dataloader:
                model.network.eval()
                x = x.float().cuda()
                y_hat = model.network(x)
                break
            working = candidate
            print(working, "works")
            if too_high == 0:
                candidate *= 2
            else:
                candidate = (candidate + too_high)//2
        except RuntimeError:
            too_high = candidate
            print(too_high, "too high")
            candidate = (candidate + working)//2
    working //= 2
    print("using batch size", working)
    torch.cuda.empty_cache()
    return working


def prediction_string(unet_out: torch.Tensor, rec: dict) -> str:
    recorder = CanvasTrafoRecorder(**rec)
    unet_out = unet_out.cpu().detach().numpy()
    unet_out_rounded = np.round(unet_out)
    boxes = BoundingBoxes.from_mask(unet_out_rounded, max_bounding_boxes=8)
    boxes = recorder.reconstruct_boxes(boxes)
    opacities = []
    for i, box in enumerate(boxes):
        if sum(box) == 0:
            break
        mask = boxes[i:i+1].get_mask(unet_out.shape)
        confidence = float(unet_out[mask].mean())
        opacities.append(" ".join(str(s)
                                  for s in ["opacity", np.round(confidence, 1),
                                            *box[:2].astype(int),
                                            *(box[2:] + box[:2]).astype(int)]))
    if len(opacities) == 0:
        return "none 1 0 0 1 1"
    return " ".join(opacities)


def get_args(*args: str) -> Namespace:
    MODEL_DIR = "_trainings/22-09_17-30_aDAOG_cB3D_b24_e200_BN_lr0.01_lrsp10/"
    parser = ArgumentParser()
    parser.add_argument("--input-dir",
                        default="_data/test_preprocessed256_rec")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--weights-filename")
    parser.add_argument("--output-dir", default="_predictions/"
                        + datetime.now().strftime('%d_%H-%M-%S'))
    parser.add_argument("--criterion", type=Criterion.__getitem__,
                        choices=Criterion, default=Criterion.B3D)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cuda-device", type=int)
    if len(args) > 0:
        return parser.parse_args(args)
    return parser.parse_args()


def main(*args: str):
    args = get_args(*args)
    SEED = 42

    assert torch.cuda.is_available(), "Missing CUDA"

    if args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data = TestData(args.input_dir)
    print("first data item:", data[0])
    with open(os.path.join(args.model_dir, "net_config.json")) as f:
        unet_config = json.load(f)

    batch_norm = unet_config["batch_norm"]
    network = Unet(batch_norm=batch_norm)
    weights_filename = args.weights_filename
    if weights_filename is None:
        weight_files = glob(os.path.join(args.model_dir, "unet_e*.ckpt"))
        weights_filename = max(weight_files, key=os.path.getctime)
        print("automatically picked latest weights", weights_filename)
        weights_filename = os.path.split(weights_filename)[-1]

    model = OurModel(name="unet", network=network,
                     criterion=args.criterion.value,
                     path_dir=args.model_dir, path_weights=weights_filename,
                     lr=0, verbose=True, segmentation=True)
    batch_size = args.batch_size or optimal_batch_size(data, model)
    dataloader = DataLoader(data, batch_size=batch_size)

    table = pd.DataFrame()

    os.makedirs(args.output_dir)
    print("output to", args.output_dir)
    with open(os.path.join(args.output_dir,
                           "image_prediction_config.json"), "w") as f:
        json.dump({"input_dir": args.input_dir, "model_dir": args.model_dir,
                   "weights_filename": weights_filename,
                   "batch_norm": batch_norm, "batch_size": batch_size,
                   "criterion": args.criterion.name}, f)

    for x, study_ids, image_ids, recs in tqdm(dataloader):
        print(study_ids, image_ids, recs)
        model.network.eval()
        x = x.float().cuda()
        y_hat = model.network(x)
        for i, (tensor, image_id) in enumerate(zip(y_hat, image_ids)):
            rec = {key: values.cpu().detach()[i]
                   for key, values in recs.items()}
            table = table.append({"Id": f"{image_id}_image",
                                  "PredictionString":
                                  prediction_string(tensor, rec)},
                                 ignore_index=True)
    table.to_csv(os.path.join(args.output_dir, "image_predictions.csv"),
                 index=False)


if __name__ == '__main__':
    main()
