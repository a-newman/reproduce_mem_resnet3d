import numbers
import os
import time
from typing import Callable, Mapping, Optional, Union

import fire
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
import train_utils as utils
from data_loader import get_dataset
from evaluate import rc
from losses import MemMSELoss
from model_mem import MemRestNet3D
from model_utils import MemModelFields, ModelOutput


def save_ckpt(savepath,
              model,
              epoch,
              it,
              optimizer,
              dataset_name,
              model_type,
              metrics=None):
    torch.save(
        {
            'epoch': epoch,
            'it': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataset_name': dataset_name,
            'model_type': model_type,
            'metrics': metrics
        }, savepath)


def main(verbose: int = 1,
         print_freq: int = 100,
         restore: Union[bool, str] = True,
         val_freq: int = 1,
         run_id: str = "model",
         dset_name: str = "memento",
         model_name: str = "resnet3d",
         freeze_until_it: int = 500,
         additional_metrics: Mapping[str, Callable] = {'rc': rc},
         debug_n: Optional[int] = None,
         batch_size: int = cfg.BATCH_SIZE,
         require_strict_model_load: bool = False,
         final_activation: str = 'relu') -> None:

    print("TRAINING MODEL {} ON DATASET {}".format(model_name, dset_name))

    ckpt_savedir = os.path.join(cfg.DATA_SAVEDIR, run_id, cfg.CKPT_DIR)
    print("Saving ckpts to {}".format(ckpt_savedir))
    logs_savepath = os.path.join(cfg.DATA_SAVEDIR, run_id, cfg.LOGDIR)
    print("Saving logs to {}".format(logs_savepath))
    utils.makedirs([ckpt_savedir, logs_savepath])
    last_ckpt_path = os.path.join(ckpt_savedir, "last_model.pth")

    device = utils.set_device()

    print('DEVICE', device)

    # model
    model = DataParallel(MemRestNet3D(final_activation=final_activation))

    if verbose:
        print(model)

    # must call this before constructing the optimizer:
    # https://pytorch.org/docs/stable/optim.html
    model.to(device)

    # set up training
    # TODO better one?
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=0.0001)
    losses = {'mem_mse': MemMSELoss()}

    initial_epoch = 0
    iteration = 0
    unfrozen = False

    if restore:
        ckpt_path = restore if isinstance(restore, str) else last_ckpt_path

        if os.path.exists(ckpt_path):

            print("Restoring weights from {}".format(ckpt_path))

            ckpt = torch.load(ckpt_path)
            utils.try_load_state_dict(model, ckpt['model_state_dict'],
                                      require_strict_model_load)
            utils.try_load_optim_state(optimizer, ckpt['optimizer_state_dict'],
                                       require_strict_model_load)
            initial_epoch = ckpt['epoch']
            iteration = ckpt['it']
    else:
        ckpt_path = last_ckpt_path

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1,
                                                   last_epoch=initial_epoch -
                                                   1)

    # dataset
    train_ds, val_ds, test_ds = get_dataset()
    assert val_ds or test_ds

    if debug_n is not None:
        train_ds = Subset(train_ds, range(debug_n))
        test_ds = Subset(test_ds, range(debug_n))

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=cfg.NUM_WORKERS)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=cfg.NUM_WORKERS)

    # training loop
    start = time.time()

    try:
        for epoch in range(initial_epoch, cfg.NUM_EPOCHS):
            logger = SummaryWriter(logs_savepath)

            # effectively puts the model in train mode.
            # Opposite of model.eval()
            model.train()

            print("Epoch {}".format(epoch))

            for i, (x, y_) in tqdm(enumerate(train_dl),
                                   total=len(train_ds) / batch_size):

                y: ModelOutput[MemModelFields] = ModelOutput(y_)
                iteration += 1

                if not unfrozen and iteration > freeze_until_it:
                    print("Unfreezing encoder")
                    unfrozen = True

                    for param in model.parameters():
                        param.requires_grad = True

                logger.add_scalar('DataTime', time.time() - start, iteration)

                x = x.to(device)
                y = y.to_device(device)

                out = ModelOutput(model(x, y.get_data()))
                loss_vals = {name: l(out, y) for name, l in losses.items()}
                loss = torch.stack(list(loss_vals.values())).sum()
                # loss = criterion(out, y)

                # I think this zeros out previous gradients (in case people
                # want to accumulate gradients?)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                utils.log_loss(logger, loss, loss_vals, iteration)
                logger.add_scalar('ItTime', time.time() - start, iteration)
                start = time.time()

                # display metrics

            # do some validation

            if (epoch + 1) % val_freq == 0:
                print("Validating...")
                model.eval()  # puts model in validation mode

                with torch.no_grad():

                    labels: Optional[ModelOutput[MemModelFields]] = None
                    preds: Optional[ModelOutput[MemModelFields]] = None
                    val_losses = []

                    for i, (x, y_) in tqdm(enumerate(test_dl),
                                           total=len(test_ds) / batch_size):

                        y = ModelOutput(y_)
                        y_numpy = y.to_numpy()

                        labels = y_numpy if labels is None else labels.merge(
                            y_numpy)

                        x = x.to(device)
                        y = y.to_device(device)

                        out = ModelOutput(model(x, y.get_data()))
                        out_numpy = out.to_device('cpu').to_numpy()
                        preds = out_numpy if preds is None else preds.merge(
                            out_numpy)

                        loss_vals = {
                            name: l(out, y)
                            for name, l in losses.items()
                        }
                        loss = torch.stack(list(loss_vals.values())).sum()
                        utils.log_loss(logger,
                                       loss,
                                       loss_vals,
                                       iteration,
                                       phase='val')

                        val_losses.append(loss)

                    print("Calculating validation metric...")
                    # print("preds", {k: v.shape for k, v in preds.items()})
                    # assert False
                    metrics = {
                        fname: f(labels, preds, losses)
                        for fname, f in additional_metrics.items()
                    }
                    print("Validation metrics", metrics)

                    for k, v in metrics.items():
                        if isinstance(v, numbers.Number):
                            logger.add_scalar('Metric_{}'.format(k), v,
                                              iteration)

                    metrics['total_val_loss'] = sum(val_losses)

                    ckpt_path = os.path.join(
                        ckpt_savedir, utils.get_ckpt_path(epoch, metrics))
                    save_ckpt(ckpt_path, model, epoch, iteration, optimizer,
                              dset_name, model_name, metrics)

            # end of epoch
            lr_scheduler.step()

            save_ckpt(last_ckpt_path, model, epoch, iteration, optimizer,
                      dset_name, model_name)

    except KeyboardInterrupt:
        print('Got keyboard interrupt, saving model...')
        save_ckpt(last_ckpt_path, model, epoch, iteration, optimizer,
                  dset_name, model_name)


if __name__ == "__main__":
    fire.Fire(main)
