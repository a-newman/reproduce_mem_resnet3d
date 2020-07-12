import torch.nn as nn

from model_utils import MemModelFields, ModelOutput


class MemMSELoss(nn.Module):
    def forward(self, y_pred: ModelOutput[MemModelFields],
                y_true: ModelOutput[MemModelFields]):
        mem_true = y_true['score']
        print("mem true", mem_true)
        # alpha_true = y_true['alpha']
        mem_pred = y_pred['score']
        print("mem pred", mem_pred)
        # alpha_pred = y_pred['alpha']

        mse_mem = nn.functional.mse_loss(mem_pred, mem_true)
        # mse_alpha = nn.functional.mse_loss(alpha_pred, alpha_true)

        return mse_mem

        # return mse_mem + mse_alpha
