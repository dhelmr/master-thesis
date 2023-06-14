import abc
from typing import Optional

from dataloader.syscall import Syscall


class SyscallFilter:
    @abc.abstractmethod
    def filter(self, syscall: Syscall) -> bool:
        raise NotImplementedError()


class MaxSyscallFilter(SyscallFilter):
    def __init__(self, max_syscalls: Optional[int] = None):
        self._max_syscalls = max_syscalls
        self._syscall_counter = 0

    def filter(self, syscall: Syscall) -> bool:
        self._syscall_counter += 1
        if self._max_syscalls is None:
            return True
        filter = self._syscall_counter <= self._max_syscalls
        return filter
