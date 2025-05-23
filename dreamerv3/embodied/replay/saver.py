import concurrent.futures
from collections import defaultdict, deque
from functools import partial as bind

from dreamerv3 import embodied
from dreamerv3.embodied.replay import chunk as chunklib


class Saver:

    def __init__(self, directory, chunks=1024):
        self.directory = embodied.Path(directory)
        self.directory.mkdirs()
        self.chunks = chunks
        self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
        self.workers = concurrent.futures.ThreadPoolExecutor(16)
        self.promises = deque()
        self.loading = False

    def add(self, step, worker):
        if self.loading:
            return
        buffer = self.buffers[worker]
        buffer.append(step)
        if buffer.length >= self.chunks:
            self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
            self.promises.append(self.workers.submit(buffer.save, self.directory))
            for promise in [x for x in self.promises if x.done()]:
                promise.result()
                self.promises.remove(promise)

    def save(self, wait=False):
        for buffer in self.buffers.values():
            if buffer.length:
                self.promises.append(self.workers.submit(buffer.save, self.directory))
        if wait:
            [x.result() for x in self.promises]
            self.promises.clear()

    def load(self, capacity, length, earliest_first=False):
        filenames = chunklib.Chunk.scan(self.directory, capacity, length - 1, earliest_first=earliest_first)
        if not filenames:
            print(f"No experience chunks in {self.directory}. This is a new replay buffer")
            return
        print(f"Loading experience chunks from {self.directory}:\n" + "\n".join(str(f) for f in filenames))
        threads = min(len(filenames), 32)
        with concurrent.futures.ThreadPoolExecutor(threads) as executor:
            chunks = list(executor.map(chunklib.Chunk.load, filenames))
        streamids = {}
        for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
            if chunk.successor not in streamids:
                streamids[chunk.uuid] = int(embodied.uuid())
            else:
                streamids[chunk.uuid] = streamids[chunk.successor]
        self.loading = True
        for i, chunk in enumerate(chunks):
            stream = streamids[chunk.uuid]
            for index in range(chunk.length):
                step = {k: v[index] for k, v in chunk.data.items()}
                yield step, stream
            # Free memory early to not require twice the replay capacity.
            chunks[i] = None
            del chunk
        self.loading = False
