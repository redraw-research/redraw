from typing import Optional

from dreamerv3.embodied.replay import generic
from dreamerv3.embodied.replay import limiters
from dreamerv3.embodied.replay import selectors


class Uniform(generic.Generic):

    def __init__(
            self, length, capacity=None, directory=None, online=False, chunks=1024,
            min_size=1, samples_per_insert=None, tolerance=1e4, seed=0,
            can_ever_add=True, can_save=True, omit_gt_state=False,
            include_in_each_sample: Optional[dict] = None,
            load_earliest_first=False, augment_images=False,
            augment_image_workers=16,
            debug_reverse_added_actions=False,
            save_framestacked_image_as_image=False,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(
                samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
            can_ever_add=can_ever_add,
            can_save=can_save,
            omit_gt_state=omit_gt_state,
            include_in_each_sample=include_in_each_sample,
            load_earliest_first=load_earliest_first,
            augment_images=augment_images,
            augment_image_workers=augment_image_workers,
            debug_reverse_added_actions=debug_reverse_added_actions,
            save_framestacked_image_as_image=save_framestacked_image_as_image,
        )
