import os
import sys
import time
import cv2
from pypylon import genicam
from pypylon import pylon

# Number of images to be grabbed.
countOfImagesToGrab = 10
output_folder = "./output_test/"

os.environ["PYLON_CAMEMU"] = "3"

# Clear output directory

for out in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, out))

print(f"Deleted all files in output directory")

# Maximum number of cameras to be used
maxCamerasToUse = 2

# The exit code of the sample application
exitCode = 0

try:
    # Get the transport layer factory
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    # Create and attach all Pylon Devices
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        cam.MaxNumBuffer = 5
        cam.Open()
        print("Using device ", cam.GetDeviceInfo().GetModelName())
        # Setting the trigger mode to On
        cam.TriggerSelector = "FrameStart"
        cam.TriggerMode.SetValue('On')

        # Setting the trigger source to software
        cam.TriggerSource.SetValue('Software')

        # Print the model name of the camera
    

    # Starts grabbing for all cameras
    for cam in cameras:
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Trigger and grab countOfImagesToGrab from the cameras
    for i in range(countOfImagesToGrab):
        for cam_idx, cam in enumerate(cameras):
            if not cam.IsGrabbing():
                break

            # Trigger the camera
            cam.ExecuteSoftwareTrigger()

            # Retrieve result from camera
            grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data
                img = grabResult.GetArray()

                # Save the image using OpenCV
                cv2.imwrite(f'{output_folder}/camera_{cam_idx}_image_{i}.jpg', img)

                # Release the grab result
                grabResult.Release()

        # Delay between image captures (if required)
        time.sleep(0.1)

except genicam.AccessException as e:
    print("An AccessException occurred: ", e)
    exitCode = 1

except genicam.GenericException as e:
    print("A GenericException occurred: ", e.GetDescription())
    exitCode = 1

sys.exit(exitCode)
