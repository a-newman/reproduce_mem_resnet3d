from collections import abc
from typing import Generic, List, Mapping, TypedDict, TypeVar, Union

import numpy as np
import torch

T = TypeVar('T', bound=Mapping)


class ModelOutput(abc.Mapping, Generic[T]):
    def __init__(self, data: T):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __repr__(self) -> str:
        return "{}(data={})".format(self.__class__.__name__, str(self.data))

    def get_data(self) -> T:
        return self.data

    @classmethod
    def _factory(cls, data: T) -> 'ModelOutput[T]':
        return cls(data=data)

    def to_numpy(self) -> 'ModelOutput[T]':
        data: T = {k: v.numpy() for k, v in self.data.items()}  # type: ignore

        return self._factory(data)

    def to_list(self) -> 'ModelOutput[T]':
        data: T = {k: v.tolist() for k, v in self.data.items()}  # type: ignore

        return self._factory(data)

    def to_device(self, device) -> 'ModelOutput[T]':
        data: T = {
            k: v.to(device)  # type: ignore
            for k, v in self.data.items()
        }

        return self._factory(data)

    def merge(self, other: 'ModelOutput[T]'):
        newdata = {
            k: np.concatenate((v, other.data[k]), axis=0)
            for k, v in self.data.items()
        }

        return self._factory(newdata)  # type: ignore


class MemModelFields(TypedDict):
    score: Union[float, torch.Tensor]
    alpha: Union[float, torch.Tensor]
