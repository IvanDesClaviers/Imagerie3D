from core.exceptions.module_exception import ModuleException


class ComputerVisionException(ModuleException):
    def __init__(self, msg):
        super(ComputerVisionException, self).__init__(msg)
