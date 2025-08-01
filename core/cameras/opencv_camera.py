import cv2
from core.cameras import Camera


class OpencvCamera(Camera):
    def __init__(self, n_cam=0, rotation=-1):
        super().__init__(rotation)
        self.id = n_cam
        self.cap = cv2.VideoCapture(self.id)
        if not self.cap.isOpened():
            print(f"Could not open Opencv camera of id {self.id}")
        print("Reading camera with opencv")
        print()

    def start_video(self):
        pass

    def stop_video(self):
        pass

    def close_camera(self):
        self.cap.release()

    def get_frame(self):
        _, f = self.cap.read()
        if self.rotation != -1:
            f = cv2.rotate(f, self.rotation)
        return f

    def is_open(self):
        return self.cap.isOpened()
