from core.exceptions.module_exception import ModuleException


class ArgumentException(ModuleException):
    def __init__(self, msg):
        super(ArgumentException, self).__init__(msg)
