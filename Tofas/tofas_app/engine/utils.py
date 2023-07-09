import cv2
import os

TEST_DIR = "/home/emir/Desktop/dev/Inovako/Inovako/Tofas/tofas_app/engine/mock_images/"
DET_CLASSES = ["DET", "NO_DET"]
class MockCameraArray:
    def __init__(self, num_cams):
        self.cameras = [MockInstantCamera(id=i) for i in range(num_cams)]
        self.is_grabbing = False

    def StartGrabbing(self, strategy=None):
        self.is_grabbing = True
        for camera in self.cameras:
            # Here you might want to initiate the grabbing process for each camera
            # but for this mock class, we'll just change a flag
            camera.is_grabbing = True

    def StopGrabbing(self):
        self.is_grabbing = False
        for camera in self.cameras:
            # Here you might want to stop the grabbing process for each camera
            # but for this mock class, we'll just change a flag
            camera.is_grabbing = False

    def __getitem__(self, index):
        return self.cameras[index]

    def __len__(self):
        return len(self.cameras)

    def Open(self):
        for camera in self.cameras:
            camera.Open()

    def Attach(self, device):
        for camera in self.cameras:
            camera.Attach(device)

class MockInstantCamera:
    def __init__(self, image_size=(720, 1280), pixel_range=(0, 255), id=0):
        self.image_size = image_size
        self.pixel_range = pixel_range
        self.PixelFormat = MockAttribute('Mono8')
        self.Width = MockAttribute(1280)
        self.Height = MockAttribute(720)
        self.TriggerSelector = "FrameStart"
        self.TriggerMode = MockAttribute('On')
        self.TriggerSource = MockAttribute('Software')
        self.ExposureTime = MockAttribute(10000)
        self.DeviceInfo = MockDeviceInfo(id)
        print(f"grabbed image {os.path.join(TEST_DIR, sorted(os.listdir(TEST_DIR))[id])} for cam: {id}")
        self.grab_result = cv2.imread(os.path.join(TEST_DIR, sorted(os.listdir(TEST_DIR))[id]))

    def Attach(self, device):
        pass

    def ExecuteSoftwareTrigger(self):
        pass

    def RetrieveResult(self, timeout, timeout_handling):
        return self.grab_result 

    def Open(self):
        pass

    def SetCameraContext(self, idx):
        pass
    def Close(self):
        pass

class MockAttribute:
    def __init__(self, value):
        self.value = value

    def SetValue(self, value):
        self.value = value

    def GetValue(self):
        return self.value

class MockDeviceInfo:
    def __init__(self, id):
        self.serial_number = f'Mock{id}'

    def GetSerialNumber(self):
        return self.serial_number

def count_images(args):
    run_id = len(os.listdir(args.out_dir))
    path = args.out_dir + f"run_{run_id}/"
    for cam_p in os.listdir(path):
        for exp in args.exposure_time:
            for d in DET_CLASSES:
                ori_p = os.path.join(os.path.join(os.path.join(path, cam_p), str(exp)), d)
                print(f"Cam: {cam_p} captured {len(os.listdir(ori_p))} images with exp: {exp}")