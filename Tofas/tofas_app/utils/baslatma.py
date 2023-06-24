from pypylon import pylon
import time
import numpy as np
import cv2
import Inovako.Tofas.tofas_app.temps.infer as infer
class BaslerCamera:
    def __init__(self, serial_number, exposure_time):
        self.camera = self.open_camera(serial_number, exposure_time)

    def open_camera(self, serial_number, exposure_time):
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera_info = pylon.DeviceInfo()
        camera_info.SetSerialNumber(serial_number)
        camera.Attach(pylon.TlFactory.GetInstance().CreateDevice(camera_info))
        camera.Open()
        camera.ExposureTime.SetValue(int(exposure_time))
        camera.MaxNumBuffer = 1
        return camera

    def set_exposure_time(self, new_value):
        self.camera.ExposureTime.SetValue(int(new_value))

    def capture_image(self):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        result = self.camera.GrabOne(500)
        image = converter.Convert(result)
        result.Release()
        return image.Array

    def close_camera(self):
        self.camera.Close()

def calculate_gray_value(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_value = np.mean(gray)
    return gray_value

def process_image(image, threshold):
    # Orta ve üst kenara odaklanan 400x400 piksellik bir alan seçimi
    height, width = image.shape[:2]
    start_x = width // 2 - 200
    end_x = width // 2 + 200
    start_y = 0
    end_y = 400

    selected_area = image[start_y:end_y, start_x:end_x]

    gray_value = calculate_gray_value(selected_area)
    if gray_value > threshold:
        result = 1
    else:
        result = 0
    print("Gray value:", gray_value)
    print("Result:", result)
    return result

def main():
    count_of_images_to_grab = 100
    threshold = 10
    serial_number = "40360112"

    try:
        
        camera = BaslerCamera(serial_number, exposure_time=10000)

        for count in range(count_of_images_to_grab):
            image = camera.capture_image()
            result = process_image(image, threshold)
            del image
            camera.close_camera()

            if result == 1:
                args = infer.parse_args()
                infer.main(args)
                # 8 Kameradan 1 i bir tanesi master kamera, Baslatma py calistiracak 24 saat calisacak
                # result 0 oldugu surece sıkıntı yok

                # result 1 = infer py calismaya basladi
                # result 0 oldu, kamera infer tarafından kullanıldıgı icin result'ın 0 oldugunu olcemiyor

            #else :
            # result 0 - tensorRT dursun

            
            time.sleep(10)

        camera.close_camera()

    except Exception as e:
        print("An exception occurred.")
        print(str(e))

if __name__ == "__main__":
    main()
