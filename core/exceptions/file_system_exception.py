from core.exceptions.module_exception import ModuleException


class FileSystemException(ModuleException):
    def __init__(self, msg):
        super(FileSystemException, self).__init__(msg)
