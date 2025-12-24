import multiprocessing

class ParallelExecutor:
    def __init__(self, func, configs: list, max_workers=None):
        self.func = func
        self.configs = configs
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def execute(self):
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            results = pool.map(self.func, self.configs)
        return results