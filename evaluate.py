import json
import os
from typing import Callable, Mapping, Optional

import fire
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config as cfg
import train_utils as utils
from data_loader import get_dataset
from model_mem import MemRestNet3D
from model_utils import MemModelFields, ModelOutput


def rc(labels: ModelOutput[MemModelFields],
       preds: ModelOutput[MemModelFields],
       _,
       verbose=True):
    mem_scores_true = labels['score']
    mem_scores_pred = preds['score']
    val = spearmanr(mem_scores_true, mem_scores_pred)

    return val.correlation


def predict(ckpt_path,
            metrics: Mapping[str, Callable] = {'rc': rc},
            num_workers: int = 20,
            use_gpu: bool = True,
            model_name: str = "resnet3d",
            dset_name: str = "memento_frames",
            batch_size: int = 1,
            preds_savepath: Optional[str] = None,
            use_val: bool = False,
            debug_n: Optional[int] = None,
            final_activation: str = 'relu',
            shuffle=False):

    print("ckpt path: {}".format(ckpt_path))

    if preds_savepath is None:
        preds_savepath = os.path.splitext(
            ckpt_path.replace(cfg.CKPT_DIR, cfg.PREDS_DIR))[0] + '.json'
        utils.makedirs([os.path.dirname(preds_savepath)])
    print("preds savepath: {}".format(preds_savepath))

    device = utils.set_device()
    print('DEVICE', device)

    # load the ckpt
    print("Loading model from path: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path)

    # model
    model = nn.DataParallel(MemRestNet3D(final_activation=final_activation))
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    model.to(device)
    model.eval()

    print("USING MODEL TYPE {} ON DSET {}".format(model_name, dset_name))

    # data loader
    train, val, test = get_dataset()
    ds = val if use_val else test

    if ds is None:
        raise ValueError("No {} set available for this dataset.".format(
            "val" if use_val else "test"))
    else:
        print("Using {} set".format("val" if use_val else "test"))

    if debug_n is not None:
        ds = Subset(ds, range(debug_n))

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)

    preds: Optional[ModelOutput] = None
    labels: Optional[ModelOutput] = None
    with torch.no_grad():
        for i, (x, y_) in tqdm(enumerate(dl), total=len(ds) / batch_size):

            y: ModelOutput[MemModelFields] = ModelOutput(y_)
            y_list = y.to_numpy()
            labels = y_list if labels is None else labels.merge(y_list)

            x = x.to(device)
            y = y.to_device(device)

            out = ModelOutput(model(x, y.get_data()))

            out_list = out.to_device('cpu').to_numpy()
            preds = out_list if preds is None else preds.merge(out_list)

    metrics = {fname: f(labels, preds, None) for fname, f in metrics.items()}
    print("METRICS", metrics)

    data = {
        'ckpt': ckpt_path,
        'preds': preds.to_list().get_data(),
        'labels': labels.to_list().get_data(),
        'metrics': metrics
    }

    with open(preds_savepath, "w") as outfile:
        print("Saving results")
        json.dump(data, outfile)

    return metrics


if __name__ == "__main__":
    fire.Fire(predict)
