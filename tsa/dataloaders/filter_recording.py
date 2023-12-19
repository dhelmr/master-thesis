from dataloader.base_recording import BaseRecording
from tsa.dataloaders.syscall_filter import SyscallFilter


class FilteredRecording(BaseRecording):
    def __init__(self, wrapped_recording: BaseRecording, filter: SyscallFilter):
        super().__init__()
        self.__wrapped_recording = wrapped_recording
        self.__filter = filter

    def syscalls(self):
        for syscall in self.__wrapped_recording.syscalls():
            if self.__filter.filter(syscall):
                yield syscall

    def packets(self):
        return  self.__wrapped_recording.packets()

    def resource_stats(self) -> list:
        return self.__wrapped_recording.resource_stats()

    def metadata(self) -> dict:
        return self.__wrapped_recording.metadata()

    def check_recording(self) -> bool:
        return self.__wrapped_recording.check_recording()
