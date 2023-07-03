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
        if self._max_syscalls is None:
            self._syscall_counter += 1
            return True

        do_filter = self._syscall_counter < self._max_syscalls
        if do_filter:
            # only increment the counter if the syscall is passed on
            # this way, the counter will never be greater than self._max_syscalls
            self._syscall_counter += 1
        return do_filter
