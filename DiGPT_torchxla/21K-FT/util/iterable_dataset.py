from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

# use synthetic dataset!
# class RandomDataset(Dataset):
#     def __init__(self, input_size, length):
#         self.input_size = input_size
#         self.length = length
#         assert isinstance(input_size, int) and input_size > 0
#         assert isinstance(length, int) and length > 0
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         image = torch.rand(3, self.input_size, self.input_size)
#         label = torch.tensor(0)
#         return image, label


class IterableImageDataset(IterableDataset):

  def __init__(
          self,
          root,
          name=None,
          reader=None,
          split='train',
          is_training=False,
          batch_size=None,
          seed=42,
          input_name='image',
          target_name='label',
          download=False,
          transform=None,
          target_transform=None,
          epoch=0,
          tokenizer=None,
          single_replica=False,
          train_num_samples=None,
  ):
    assert reader is not None
    # if isinstance(reader, str):
    #   self.reader = create_reader(
    #     reader,
    #     root=root,
    #     split=split,
    #     is_training=is_training,
    #     batch_size=batch_size,
    #     seed=seed,
    #     repeats=repeats,
    #     download=download,
    #   )
    # else:
    #   self.reader = reader
    # only support tfds for now
    from .reader_tfds import ReaderTfds  # defer tensorflow import
    self.reader = ReaderTfds(root=root,
                             name=name,
                             split=split,
                             is_training=is_training,
                             batch_size=batch_size,
                             seed=seed,
                             input_name=input_name,
                             target_name=target_name,
                             download=download,
                             epoch=epoch,
                             single_replica=single_replica,
                             train_num_samples=train_num_samples,
                             )
    self.transform = transform
    self.target_transform = target_transform
    self.tokenizer = tokenizer
    assert not (self.target_transform is not None and self.tokenizer is not None)
    self._consecutive_errors = 0

  def __iter__(self):
    for img, target in self.reader:
      if self.transform is not None:
        img = self.transform(img)
      if self.target_transform is not None:
        target = self.target_transform(target)
      if self.tokenizer is not None:
        target = self.tokenizer(str(target))[0]
      # if isinstance(img, (tuple, list)):
      #   img, patch_indices_keep = img
      #   yield img, patch_indices_keep, target
      # else:
      yield img, target

  def __len__(self):
    if hasattr(self.reader, '__len__'):
      return len(self.reader)
    else:
      return 0

  def set_epoch(self, count):
    # TFDS and WDS need external epoch count for deterministic cross process shuffle
    if hasattr(self.reader, 'set_epoch'):
      self.reader.set_epoch(count)

  def set_loader_cfg(
          self,
          num_workers: Optional[int] = None,
  ):
    # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
    if hasattr(self.reader, 'set_loader_cfg'):
      self.reader.set_loader_cfg(num_workers=num_workers)

  def filename(self, index, basename=False, absolute=False):
    assert False, 'Filename lookup by index not supported, use filenames().'

  def filenames(self, basename=False, absolute=False):
    return self.reader.filenames(basename, absolute)


def get_iterable_dataloader(args, root, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert root
    split = 'train'
    name = None
    #name = 'imagenet2012:5.1.0'
    #name = 'imagenet21k/tfrecord:1.0.0'
    input_name = 'image'
    target_name = 'label'

    dataset = IterableImageDataset(
        root,
        name=name,
        reader='tfds',
        split=split,
        is_training=is_train,
        batch_size=args.batch_size,
        seed=args.seed,
        input_name=input_name,
        target_name=target_name,
        transform=preprocess_fn,
        target_transform=None,
        epoch=epoch,
        tokenizer=tokenizer,
        single_replica=not is_train,
    )
    # give Iterable datasets early knowledge of num_workers so that sample estimates
    # are correct before worker processes are launched
    dataset.set_loader_cfg(num_workers=args.num_workers if args.num_workers != 0 else 1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # tfds has a internal shuffling mechanism
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        # persistent_workers=False,
    )

    dataloader.num_samples = len(dataset) * args.world_size
    dataloader.num_batches = len(dataloader)

    return dataloader