import time
from collections import defaultdict, deque
from functools import partial as bind
from typing import Optional
import multiprocessing as mp
import threading
import numpy as np
from typing import List
# import albumentations as A
import blosc

from dreamerv3 import embodied
from dreamerv3.embodied.replay import saver

#
# def _augment_images(input_queue, processed_queue):
#     augment_image = A.Compose([
#         A.ISONoise(),
#         A.Superpixels(n_segments=100, max_size=128, interpolation=1, p=0.7),
#         A.CoarseDropout(max_holes=3, max_height=40, max_width=40, min_holes=0, min_height=1, min_width=1, fill_value=0, p=0.3),
#         A.Sharpen(p=0.1),
#         A.PixelDropout(dropout_prob=0.005, p=0.5),
#         A.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, p=0.8),
#         A.RandomFog(p=0.5),
#         A.MotionBlur(blur_limit=9, p=0.5),
#         A.OpticalDistortion(p=0.5),
#         A.RandomBrightnessContrast(contrast_limit=0.0),
#         A.ISONoise(),
#     ])
#     while True:
#         seq = input_queue.get()
#         if seq is None:
#             continue
#         seq.update({
#             "original_image": seq['image'],
#             "image": [augment_image(image=im)['image'] for im in seq['image']]
#         })
#         seq = {k: embodied.convert(v) for k, v in seq.items()}
#         processed_queue.put(seq)
#
# def _augment_images2(input_queue, processed_queue):
#     transform1 = A.Compose([
#         A.ISONoise(),
#         A.ChannelDropout(p=0.1),
#         A.CoarseDropout(max_holes=3, max_height=40, max_width=40, min_holes=0, min_height=1, min_width=1,
#                         fill_value="random", p=0.3)
#     ])
#     transform2 = A.Compose([
#         A.Superpixels(p_replace=(0.0, 0.3), n_segments=100, max_size=256, interpolation=1, p=0.7),
#         A.Sharpen(p=0.1),
#         A.PixelDropout(dropout_prob=0.005, p=0.5),
#         A.ColorJitter(brightness=0.3, contrast=0.3, hue=0.15, p=0.8),
#         A.RandomFog(p=0.5),
#         A.MotionBlur(blur_limit=9, p=0.5),
#         A.OpticalDistortion(p=0.5),
#         A.RandomToneCurve(scale=0.2, per_channel=False, p=1),
#         A.ISONoise(p=0.1),
#     ])
#     while True:
#         seq = input_queue.get()
#         if seq is None:
#             continue
#         seq.update({
#             "original_image": seq['image'],
#             "image": [transform2(**transform1(image=im))['image'] for im in seq['image']]
#         })
#         seq = {k: embodied.convert(v) for k, v in seq.items()}
#         processed_queue.put(seq)
#
#
# def _augment_images_mask(input_queue, processed_queue):
#     from duckiebots_unreal_sim.rcan_obs_preprocessor import ONNXRCANObsPreprocessor
#
#     transform1 = A.Compose([
#         A.CoarseDropout(max_holes=2, max_height=20, max_width=20, min_holes=0, min_height=1, min_width=1, p=0.3),
#         A.Superpixels(p_replace=(0.0, 0.2), n_segments=50, max_size=64, interpolation=1, p=0.4),
#         A.OpticalDistortion(p=0.3),
#         A.MotionBlur(blur_limit=9, p=0.4),
#     ])
#
#     rcan_checkpoint_path = "/home/author1/Downloads/ckpt_9_nov_17.onnx"
#     rcan = ONNXRCANObsPreprocessor(checkpoint_path=rcan_checkpoint_path,
#                                    debug_render_predictions=False)
#     rcan_transform2 = lambda rgb_obs: rcan.preprocess_obs(rgb_obs=rgb_obs)
#
#     while True:
#         seq = input_queue.get()
#         if seq is None:
#             continue
#         seq.update({
#             "original_image": seq['image'],
#             "image": rcan_transform2([transform1(image=im)['image'] for im in seq['image']])
#         })
#         seq = {k: embodied.convert(v) for k, v in seq.items()}
#         processed_queue.put(seq)

# def _augment_images_mask(input_queue, processed_queue):
#     from dreamerv3.embodied.replay.mask_augmentation.transforms import SuperpixelsDuckiebotsMask
#     transform1 = A.Compose([
#         A.CoarseDropout(max_holes=2, max_height=20, max_width=20, min_holes=0, min_height=1, min_width=1, p=0.5),
#         SuperpixelsDuckiebotsMask(max_size=128, p=0.5),
#     ])
#     while True:
#         seq = input_queue.get()
#         if seq is None:
#             continue
#         seq.update({
#             "original_image": seq['image'],
#             "image": [transform1(image=im)['image'] for im in seq['image']]
#         })
#         seq = {k: embodied.convert(v) for k, v in seq.items()}
#         processed_queue.put(seq)

class Generic:

    def __init__(
            self, length, capacity, remover, sampler, limiter, directory,
            overlap=None, online=False, chunks=1024,
            can_ever_add=True, can_save=True, omit_gt_state=False,
            include_in_each_sample: Optional[dict] = None,
            load_earliest_first=False,
            augment_images=None,
            augment_image_workers=16,
            augment_images_sample_queue_depth=16*8,
            debug_reverse_added_actions=False,
            save_framestacked_image_as_image=False
    ):
        assert capacity is None or 1 <= capacity
        self.length = length
        self.capacity = capacity
        self.remover = remover
        self.sampler = sampler
        self.limiter = limiter
        self.stride = 1 if overlap is None else length - overlap
        self.streams = defaultdict(bind(deque, maxlen=length))
        self.counters = defaultdict(int)
        self.table = {}
        self.online = online
        if self.online:
            self.online_queue = deque()
            self.online_stride = length
            self.online_counters = defaultdict(int)
        self.can_ever_add = can_ever_add
        self.can_save = can_save
        self.saver = directory and saver.Saver(directory, chunks)
        self.metrics = {
            'samples': 0,
            'sample_wait_dur': 0,
            'sample_wait_count': 0,
            'inserts': 0,
            'insert_wait_dur': 0,
            'insert_wait_count': 0,
        }
        self._debug_reverse_added_actions = debug_reverse_added_actions
        self._image_compression_spec = None  # will hold (shape, dtype) after first add
        self._save_framestacked_image_as_image = save_framestacked_image_as_image
        if self._save_framestacked_image_as_image:
            print(f"saving framestacked_image as image!")

        self.load_earliest_first = load_earliest_first
        self.load()
        self.omit_gt_state = omit_gt_state
        self.include_in_each_sample = include_in_each_sample

        self._threads: List[threading.Thread] = []
        self._running = True
        self._load_samples_thread = None
        self.augment_images = augment_images
        self._start_load_samples_lock = threading.Lock()
        if bool(augment_images):
            self._unprocessed_samples_queue = mp.Queue(maxsize=augment_images_sample_queue_depth)
            self._processed_samples_queue = mp.Queue(maxsize=augment_images_sample_queue_depth)
            self._load_samples_thread = threading.Thread(
                target=self._load_samples_for_processing,
                daemon=True)
            self._threads.append(self._load_samples_thread)


            # if augment_images == "v1":
            #     worker_target = _augment_images
            # elif augment_images == "v2":
            #     worker_target = _augment_images2
            # elif augment_images == "mask":
            #     worker_target = _augment_images_mask
            # else:
            raise ValueError(f"Unknown value for augment images: {augment_images}")

            workers = [mp.Process(target=worker_target, args=(self._unprocessed_samples_queue, self._processed_samples_queue)) for _ in range(augment_image_workers)]
            for worker in workers:
                worker.daemon = True
                worker.start()
    
    def _compress_image(self, arr):
        """Encode uint8 image array → bytes (lossless)."""
        if self._image_compression_spec is None:              # learn spec first time
            self._image_spec = (arr.shape, arr.dtype)
        shape, dtype = self._image_spec
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(f"Image spec mismatch: expected {shape}/{dtype}, "
                             f"got {arr.shape}/{arr.dtype}")
        return blosc.compress(
            arr.tobytes(),
            typesize=1, cname="zstd", clevel=3, shuffle=blosc.BITSHUFFLE,
        )

    def _decompress_image(self, buf):
        """bytes → uint8 image array with recorded shape."""
        shape, dtype = self._image_spec
        return np.frombuffer(blosc.decompress(buf), dtype=dtype).reshape(shape)
    
    def _load_samples_for_processing(self):
        while self._running:
            sample = self._sample()
            self._unprocessed_samples_queue.put(sample, block=True)

    def __len__(self):
        return len(self.table)

    @property
    def stats(self):
        ratio = lambda x, y: x / y if y else np.nan
        m = self.metrics
        stats = {
            'size': len(self),
            'inserts': m['inserts'],
            'samples': m['samples'],
            'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
            'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
            'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
            'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
        }
        for key in self.metrics:
            self.metrics[key] = 0
        return stats

    def add(self, step, worker=0, load=False):
        if not (self.can_ever_add or load):
            raise ValueError("This replay buffer is frozen and cannot be added to.")
        step = {k: v for k, v in step.items() if not k.startswith('log_')}
        if self._debug_reverse_added_actions:
            assert isinstance(step['action'], np.ndarray), type(step['action'])
            step['action'] = -step['action']
        step['id'] = np.asarray(embodied.uuid(step.get('id')))
        stream = self.streams[worker]
        if self.saver and self.can_save:
            if self._save_framestacked_image_as_image:
                save_step = step.copy()
                save_step['image'] = save_step['framestacked_image']
                del save_step['framestacked_image']
            else:
                save_step = step
            self.saver.add(save_step, worker)
        if "image" in step:
            step = dict(step) # shallow copy
            step["image"] = self._compress_image(step["image"])
        stream.append(step)
        self.counters[worker] += 1
        if self.online:
            self.online_counters[worker] += 1
            if len(stream) >= self.length and (
                    self.online_counters[worker] >= self.online_stride):
                self.online_queue.append(tuple(stream))
                self.online_counters[worker] = 0
        if len(stream) < self.length or self.counters[worker] < self.stride:
            return
        self.counters[worker] = 0
        key = embodied.uuid()
        seq = tuple(stream)
        if load:
            assert self.limiter.want_load()[0]
        else:
            dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
            self.metrics['inserts'] += 1
            self.metrics['insert_wait_dur'] += dur
            self.metrics['insert_wait_count'] += int(dur > 0)
        self.table[key] = seq
        self.remover[key] = seq
        self.sampler[key] = seq
        while self.capacity and len(self) > self.capacity:
            self._remove(self.remover())

    def _sample(self):
        if not self.limiter.want_sample(stateless=True):
            dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
            self.metrics['sample_wait_dur'] += dur
            self.metrics['sample_wait_count'] += int(dur > 0)
        self.metrics['samples'] += 1
        if self.online:
            try:
                seq = self.online_queue.popleft()
            except IndexError:
                seq = self.table[self.sampler()]
        else:
            seq = self.table[self.sampler()]
        seq_len = len(seq)
        
        # Unpack each step, decompressing images ------------------------
        seq_unpacked = []
        for st in seq:
           if "image" in st:
               st = dict(st) # shallow copy
               st["image"] = self._decompress_image(st["image"])
           seq_unpacked.append(st)

        seq = {k: [step[k] for step in seq_unpacked] for k in seq_unpacked[0]}
        # seq = {k: [step[k] for step in seq] for k in seq[0]}
        if self.include_in_each_sample:
            seq.update({k: [v for _ in range(seq_len)] for k, v in self.include_in_each_sample.items()})
        seq = {k: embodied.convert(v) for k, v in seq.items()}
        if 'is_first' in seq:
            seq['is_first'][0] = True
        # Setting 'is_last' added for SimDreamer:
        if 'is_last' in seq:
            seq['is_last'][-1] = True
        if self.omit_gt_state and 'gt_state' in seq:
            del seq['gt_state']
        return seq

    def _remove(self, key):
        wait(self.limiter.want_remove, 'Replay remove is waiting')
        del self.table[key]
        del self.remover[key]
        del self.sampler[key]

    def dataset(self):
        if self.augment_images:
            with self._start_load_samples_lock:
                if not self._load_samples_thread.is_alive():
                    self._load_samples_thread.start()
            while True:
                yield self._processed_samples_queue.get()
        else:
            while True:
                yield self._sample()

    def prioritize(self, ids, prios):
        if hasattr(self.sampler, 'prioritize'):
            self.sampler.prioritize(ids, prios)

    def save(self, wait=False):
        if not self.saver or not self.can_save:
            return
        self.saver.save(wait)
        # return {
        #     'saver': self.saver.save(wait),
        #     # 'remover': self.remover.save(wait),
        #     # 'sampler': self.sampler.save(wait),
        #     # 'limiter': self.limiter.save(wait),
        # }

    def load(self, data=None):
        if not self.saver:
            return
        workers = set()
        for step, worker in self.saver.load(self.capacity, self.length, earliest_first=self.load_earliest_first):
            workers.add(worker)
            self.add(step, worker, load=True)
        for worker in workers:
            del self.streams[worker]
            del self.counters[worker]
        # self.remover.load(data['remover'])
        # self.sampler.load(data['sampler'])
        # self.limiter.load(data['limiter'])
    def load_steps_in_order(self):
        # hasnt been optimized for memory efficiency, but it may be fine for our use case
        steps_to_load = deque(maxlen=self.capacity)
        for step, worker in self.saver.load(self.capacity, self.length, earliest_first=self.load_earliest_first):
            steps_to_load.append((step, worker))

        sequential_step_streams = defaultdict(list)
        for step, worker in steps_to_load:
            sequential_step_streams[worker].append(step)

        combined_step_stream = []
        for step_stream in sequential_step_streams.values():
            if 'is_first' in step_stream:
                step_stream[0]['is_first'] = True
            # Setting 'is_last' added for SimDreamer:
            if 'is_last' in step_stream:
                step_stream[-1]['is_last'] = True
            combined_step_stream.extend(step_stream)
        return combined_step_stream

    def all_steps_iterator(self):
        for step, worker in self.saver.load(self.capacity, self.length, earliest_first=self.load_earliest_first):
            yield {k: embodied.convert(v) for k, v in step.items()}

    def close(self):
        self._running = False
        for thread in self._threads:
            thread.close()
def wait(predicate, message, sleep=0.001, notify=1.0):
    start = time.time()
    notified = False
    while True:
        allowed, detail = predicate()
        duration = time.time() - start
        if allowed:
            return duration
        if not notified and duration >= notify:
            print(f'{message} ({detail})')
            notified = True
        time.sleep(sleep)
