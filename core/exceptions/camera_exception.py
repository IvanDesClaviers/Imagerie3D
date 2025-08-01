from core.exceptions.module_exception import ModuleException


class CameraException(ModuleException):
    def __init__(self, msg):
        super(CameraException, self).__init__(msg)
