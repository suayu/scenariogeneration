"""在隔离STRIVE worker内按墙钟预算结束完整迭代，不修改官方源码。"""
import time
from types import SimpleNamespace


class OptimizationBudget:
    """保留已完成迭代的参数；禁止在梯度更新中途截取半成品。"""

    def __init__(self, seconds, clock=time.monotonic):
        if seconds <= 0:
            raise ValueError('optimization budget must be positive')
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + seconds
        self.iterations = {}
        self.truncated_stages = []

    @property
    def expired(self):
        return self.clock() >= self.deadline

    def install(self, function, stage):
        """仅替换函数模块的进度条迭代器；损失、Adam和最终解码保持官方实现。"""
        original = function.__globals__['tqdm']
        budget = self
        budget.iterations[stage] = 0

        class Progress:
            def __init__(self, iterable, *args, **kwargs):
                self.bar = original.tqdm(iterable, *args, **kwargs)

            def __iter__(self):
                for item in self.bar:
                    if budget.expired:
                        budget.truncated_stages.append(stage)
                        self.bar.close()
                        break
                    yield item
                    budget.iterations[stage] += 1

            def __getattr__(self, key):
                return getattr(self.bar, key)

        function.__globals__['tqdm'] = SimpleNamespace(tqdm=Progress)

    def audit(self):
        return {'completed_iterations': dict(self.iterations),
                'truncated_stages': list(self.truncated_stages),
                'optimization_elapsed_seconds': self.clock() - self.started,
                'budget_semantics': 'stop_before_next_iteration; finish_current_iteration_and_final_decode'}
