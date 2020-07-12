import random

import torch
from torchvision import transforms as T

import config as cfg
from datasets import (MementoMemAlphaLabelSet, MementoRecordSet,
                      VideoRecordLoader)
from model_utils import ModelOutput
from torchvideo.samplers import FrameSampler
from torchvideo.transforms import (CenterCropVideo, CollectFrames,
                                   PILVideoToTensor, RandomCropVideo,
                                   ResizeVideo, TimeToChannel, Transform)


class RescaleInRange(Transform):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        assert self.lower < self.upper

    def _gen_params(self, frames):
        return None

    def _transform(self, frames, params):
        maxval = torch.max(frames)
        minval = torch.min(frames)
        spread = self.upper - self.lower
        current_spread = maxval - minval

        return (frames - minval) * (spread / current_spread) + self.lower


class ApplyToKeysTransform(object):
    """Applies transforms to the keys of a ModelOutput obj"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample: ModelOutput):
        data = {k: self.transform([v])[0] for k, v in sample.items()}

        return ModelOutput(data)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.transform))


IMAGE_TRAIN_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.RandomCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
IMAGE_TEST_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.CenterCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
Y_TRANSFORMS = ApplyToKeysTransform(torch.FloatTensor)
# Y_TRANSFORMS = None
VIDEO_TRAIN_TRANSFORMS = T.Compose([
    ResizeVideo(cfg.RESIZE),
    RandomCropVideo(cfg.CROP_SIZE),
    CollectFrames(),
    PILVideoToTensor(rescale=True),
    RescaleInRange(0, 1)
])
VIDEO_TEST_TRANSFORMS = T.Compose([
    ResizeVideo((cfg.RESIZE, cfg.RESIZE)),
    CenterCropVideo((cfg.CROP_SIZE, cfg.CROP_SIZE)),
    CollectFrames(),
    PILVideoToTensor(rescale=True),
    RescaleInRange(0, 1)
])
FRAMES_TRAIN_TRANSFORMS = T.Compose([VIDEO_TRAIN_TRANSFORMS, TimeToChannel()])
FRAMES_TEST_TRANSFORMS = T.Compose([VIDEO_TEST_TRANSFORMS, TimeToChannel()])


def get_dataset():
    train_ds = get_memento_video_loader(split="train",
                                        transform=VIDEO_TRAIN_TRANSFORMS,
                                        target_transform=Y_TRANSFORMS)
    val_ds = get_memento_video_loader(split="val",
                                      transform=VIDEO_TEST_TRANSFORMS,
                                      target_transform=Y_TRANSFORMS)
    test_ds = get_memento_video_loader(split="test",
                                       transform=VIDEO_TEST_TRANSFORMS,
                                       target_transform=Y_TRANSFORMS)

    return train_ds, val_ds, test_ds


class NRandomFramesSampler(FrameSampler):
    def __init__(self, nframes):
        self.nframes = nframes

    def sample(self, video_length):
        if video_length < 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".
                format(video_length))
        indices = []

        while len(indices) < self.nframes:
            indices.extend(list(range(video_length - 1)))

        indices = sorted(random.sample(indices, k=self.nframes))

        return indices


class NFramesSampler(FrameSampler):
    def __init__(self, nframes, avoid_final_frame=True):
        self.nframes = nframes
        self.avoid_final_frame = avoid_final_frame

    def sample(self, video_length):
        if self.avoid_final_frame:
            video_length = video_length - 1

        if video_length == 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".
                format(video_length))

        indices = [
            i * video_length // self.nframes + video_length //
            (2 * self.nframes) for i in range(self.nframes)
        ]

        return indices

    def __repr__(self):
        return self.__class__.__name__ + "(nframes={})".format(self.nframes)


def get_memento_video_loader(split,
                             metadata_path=cfg.MEMENTO_METADATA_PATH,
                             transform=None,
                             target_transform=None):
    sampler = NFramesSampler(nframes=45)
    record_set = MementoRecordSet.from_metadata_file()

    label_set = MementoMemAlphaLabelSet(split=split, factor=100)
    filter_func = lambda r: label_set.is_in_set(r.filename)
    vidloader = VideoRecordLoader(record_set=record_set,
                                  label_set=label_set,
                                  filter=filter_func,
                                  sampler=sampler,
                                  transform=transform,
                                  target_transform=target_transform)

    return vidloader
