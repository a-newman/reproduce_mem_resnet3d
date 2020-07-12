import sys

import torch
import torch.nn as nn

from model_utils import MemModelFields

try:
    sys.path.append("threed_resnets_pytorch")
    from model import generate_model, load_pretrained_model
    from opts import get_parser
except:
    raise RuntimeError()


class MemRestNet3D(nn.Module):
    def __init__(self,
                 model_name="resnet",
                 depth=34,
                 freeze_encoder=True,
                 final_activation='tanh'):
        super(MemRestNet3D, self).__init__()
        assert final_activation == 'tanh' or final_activation == 'relu'
        self.model_name = model_name
        self.depth = depth
        self.pretrained_path = "r3d34_K_200ep.pth"

        self.base = generate_model(self.get_opts(self.model_name, self.depth))
        self.base = load_pretrained_model(self.base,
                                          pretrain_path=self.pretrained_path,
                                          model_name=self.model_name,
                                          n_finetune_classes=2)
        self.final_activation = nn.LeakyReLU(
        ) if final_activation == 'relu' else nn.Tanh()

        if freeze_encoder:
            for param in self.base.parameters():
                param.requires_grad = False

            for param in self.base.fc.parameters():
                param.requires_grad = True

    @staticmethod
    def get_opts(model_name, depth):
        """Kind of a janky way to get the model correctly configured"""
        model_name = "resnet"
        args = [
            "--model", model_name, "--model_depth",
            str(depth), "--n_classes",
            str(700)
        ]
        opt = get_parser().parse_args(args)
        opt.n_input_channels = 3

        return opt

    def forward(self, x: torch.Tensor, _) -> MemModelFields:
        features = self.base(x)

        out = self.final_activation(features)

        mem_scores = out[:, 0]
        alphas = out[:, 1]
        data: MemModelFields = {'score': mem_scores, 'alpha': alphas}

        return data
