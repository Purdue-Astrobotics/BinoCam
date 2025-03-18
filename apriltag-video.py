#!/usr/bin/env python

from argparse import ArgumentParser
import cv2
import apriltag
import numpy as np

################################################################################


def apriltag_video(
    camera_index=0,  # 0 for built-in webcam, 1 for external webcam
    display_stream=True,
    detection_window_name="AprilTag",
    tag_size=0.0762,  # Real-world size of the tag in meters (adjust as needed)
    camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909),  # fx, fy, cx, cy
):
    """
    Detect AprilTags from real-time webcam feed and compute depth.

    Args:
        camera_index [int]: Index of the webcam to use
        display_stream [bool]: Boolean flag to display/not stream annotated with detections
        detection_window_name [str]: Title of displayed (output) tag detection window
        tag_size [float]: Size of the AprilTag in meters
        camera_params [tuple]: Camera intrinsic parameters (fx, fy, cx, cy)
    """

    parser = ArgumentParser(description="Detect AprilTags from real-time video stream.")
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    """
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    """

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = video.read()
        if not success:
            break

        # Convert to grayscale for apriltag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        for detection in detections:
            # Get pose estimation
            pose, e0, e1 = detector.detection_pose(detection, camera_params, tag_size)

            # Extract depth (distance from camera)
            depth = pose[2, 3]  # t_z value from transformation matrix

            # Draw depth information on frame
            tag_center = tuple(map(int, detection.center))
            cv2.putText(
                frame,
                f"Depth: {depth:.2f}m",
                (tag_center[0] - 50, tag_center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw detection box
            for i in range(4):
                p1 = tuple(map(int, detection.corners[i]))
                p2 = tuple(map(int, detection.corners[(i + 1) % 4]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

        if display_stream:
            cv2.imshow(detection_window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to terminate
                break

    video.release()
    cv2.destroyAllWindows()


################################################################################

if __name__ == "__main__":
    apriltag_video()
