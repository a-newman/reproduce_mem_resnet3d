# Training a ResNet3D on Memento

As a baseline for the Memento paper, replicating Section 5.2 of the [VideoMem paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cohendet_VideoMem_Constructing_Analyzing_Predicting_Short-Term_and_Long-Term_Video_Memorability_ICCV_2019_paper.pdf)

Model courtesy of [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) (configured as submodule).

## Training 

To train a ResNet3D-34 on Memento, run (for example):

```
python3 train.py --run_id "resnet3d_no_alpha" --batch_size 23 --freze_until_it 750
```

See `train.py --help` for more flags.

## Evaluating 

Run: 

```
python3 evaluate.py --ckpt_path data/resnet3d_no_alpha/ckpt/ep_9__rc_0.57359067728224__total_val_loss_3625.34521484375.pth --batch_size 23
```

This will reproduce the number reported in the Memento paper. Output from this command is recorded in `test_results.txt`.

See `evaluate.py --help` for more flags.
