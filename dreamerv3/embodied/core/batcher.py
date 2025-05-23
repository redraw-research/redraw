import queue as queuelib
import sys
import threading
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import itertools
import collections

class Batcher:

    def __init__(
            self, sources, workers=0, postprocess=None,
            prefetch_source=4, prefetch_batch=2):
        self._workers = workers
        self._postprocess = postprocess
        if workers:
            # Round-robin assign sources to workers.
            self._running = True
            self._threads = []
            self._queues = []
            assignments = [([], []) for _ in range(workers)]
            for index, source in enumerate(sources):
                queue = queuelib.Queue(prefetch_source)
                self._queues.append(queue)
                assignments[index % workers][0].append(source)
                assignments[index % workers][1].append(queue)
            for args in assignments:
                creator = threading.Thread(
                    target=self._creator, args=args, daemon=True)
                creator.start()
                self._threads.append(creator)
            self._batches = queuelib.Queue(prefetch_batch)
            batcher = threading.Thread(
                target=self._batcher, args=(self._queues, self._batches),
                daemon=True)
            batcher.start()
            self._threads.append(batcher)
        else:
            self._iterators = [source() for source in sources]
            self._prefetch_amt = prefetch_batch
            self._batch_iterator = self._get_local_worker_batch_iter()
            self._batch_prefetch_iterator = self._get_local_worker_prefetch_batch_iter()
            self._batch_prefetch_queue = collections.deque()
            # def _prefetch_loop():
            #     # exception_ever_raised = False
            #     while True:
            #         try:
            #             # print(f"fetch batch exception_ever_raised={exception_ever_raised}")
            #             self.fetch_batch()
            #         except ValueError as e:
            #             print(f"not raising error {type(e)} {e}")
            #             # exception_ever_raised = True
            #         except Exception as e:
            #             print(f"raising error {type(e)} {e}")
            #             # exception_ever_raised = True
            #             raise e
            #         time.sleep(0.01)
            # self._prefetch_thread = threading.Thread(target=_prefetch_loop)
            # self._prefetch_thread.start()

        self._once = False

    def _get_local_worker_batch_iter(self):
        while True:
            elems = [next(x) for x in self._iterators]
            batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
            if self._postprocess:
                batch = self._postprocess(batch)
            if isinstance(batch, Exception):
                raise batch
            yield batch

    def _local_worker_fetch_batch(self, n_batch):
        for batch in itertools.islice(self._batch_iterator, n_batch):
            self._batch_prefetch_queue.append(batch)

    def _get_local_worker_prefetch_batch_iter(self):
        while True:
            if self._batch_prefetch_queue:
                yield self._batch_prefetch_queue.popleft()
                self.fetch_batch()
            else:
                self.fetch_batch()

            #     assert self._prefetch_thread.is_alive()
            #     time.sleep(0.001)
    def fetch_batch(self):
        # print(f"fetch batch call len(self._batch_prefetch_queue) = {len(self._batch_prefetch_queue)} self._prefetch_amt={self._prefetch_amt} {time.time()}")
        if not self._workers and len(self._batch_prefetch_queue) < self._prefetch_amt:
            # print(f"fetch batch trigger {time.time()}")
            self._local_worker_fetch_batch(1)

    def close(self):
        if self._workers:
            self._running = False
            for thread in self._threads:
                thread.close()

    def __iter__(self):
        if self._once:
            raise RuntimeError(
                'You can only create one iterator per Batcher object to ensure that '
                'data is consumed in order. Create another Batcher object instead.')
        self._once = True
        return self

    def __call__(self):
        return self.__iter__()

    def __next__(self):
        if self._workers:
            batch = self._batches.get()
            if isinstance(batch, Exception):
                raise batch
        else:
            batch = next(self._batch_prefetch_iterator)
        return batch

    # @staticmethod
    # @partial(jax.jit, static_argnames=['postprocess'])
    # def _next_prefetch_local_worker(elems, postprocess=None):
    #     print("Tracing _next_prefetch_local_worker")
    #     batch = {k: jnp.stack([x[k] for x in elems], 0) for k in elems[0]}
    #     if postprocess:
    #         batch = postprocess(batch)
    #     return batch

    def _creator(self, sources, outputs):
        try:
            iterators = [source() for source in sources]
            while self._running:
                waiting = True
                for iterator, queue in zip(iterators, outputs):
                    if queue.full():
                        continue
                    queue.put(next(iterator))
                    waiting = False
                if waiting:
                    time.sleep(0.001)
        except Exception as e:
            e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            outputs[0].put(e)
            raise

    def _batcher(self, sources, output):
        try:
            while self._running:
                elems = [x.get() for x in sources]
                for elem in elems:
                    if isinstance(elem, Exception):
                        raise elem
                batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
                if self._postprocess:
                    batch = self._postprocess(batch)
                output.put(batch)  # Will wait here if the queue is full.
        except Exception as e:
            e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            output.put(e)
            raise





class MultiProcessBatcher:
    # TODO

    def __init__(
            self, sources, workers=0, postprocess=None,
            prefetch_source=4, prefetch_batch=2):
        raise NotImplementedError
        self._workers = workers
        self._postprocess = postprocess
        if workers:
            # Round-robin assign sources to workers.
            self._running = True
            self._threads = []
            self._queues = []
            assignments = [([], []) for _ in range(workers)]
            for index, source in enumerate(sources):
                queue = queuelib.Queue(prefetch_source)
                self._queues.append(queue)
                assignments[index % workers][0].append(source)
                assignments[index % workers][1].append(queue)
            for args in assignments:
                creator = threading.Thread(
                    target=self._creator, args=args, daemon=True)
                creator.start()
                self._threads.append(creator)
            self._batches = queuelib.Queue(prefetch_batch)
            batcher = threading.Thread(
                target=self._batcher, args=(self._queues, self._batches),
                daemon=True)
            batcher.start()
            self._threads.append(batcher)
        else:
            self._iterators = [source() for source in sources]
        self._once = False

    def close(self):
        if self._workers:
            self._running = False
            for thread in self._threads:
                thread.close()

    def __iter__(self):
        if self._once:
            raise RuntimeError(
                'You can only create one iterator per Batcher object to ensure that '
                'data is consumed in order. Create another Batcher object instead.')
        self._once = True
        return self

    def __call__(self):
        return self.__iter__()

    def __next__(self):
        if self._workers:
            batch = self._batches.get()
        else:
            elems = [next(x) for x in self._iterators]
            batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
        if isinstance(batch, Exception):
            raise batch
        return batch

    def _creator(self, sources, outputs):
        try:
            iterators = [source() for source in sources]
            while self._running:
                waiting = True
                for iterator, queue in zip(iterators, outputs):
                    if queue.full():
                        continue
                    queue.put(next(iterator))
                    waiting = False
                if waiting:
                    time.sleep(0.001)
        except Exception as e:
            e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            outputs[0].put(e)
            raise

    def _batcher(self, sources, output):
        try:
            while self._running:
                elems = [x.get() for x in sources]
                for elem in elems:
                    if isinstance(elem, Exception):
                        raise elem
                batch = {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}
                if self._postprocess:
                    batch = self._postprocess(batch)
                output.put(batch)  # Will wait here if the queue is full.
        except Exception as e:
            e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            output.put(e)
            raise

