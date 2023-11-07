from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath(os.path.join(
#     exec_path, "retinanet.pth")
# )
# detector.loadModel()
#
# list = detector.detectObjectsFromImage(
#     input_image=os.path.join(exec_path, "1.jpg"),
#     output_image_path=os.path.join(exec_path, "new_objects.jpg")
#     # minimum_percentage_probability=90,
#     # display_percentage_probability=True,
#     # display_object_name=True
# )

from imageai.Detection import VideoObjectDetection

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exec_path, "yolov3.pt"))
detector.loadModel()
detector.useCPU()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(exec_path, "1.mp4"),
    output_file_path=os.path.join(exec_path, "traffic_detected"),
    frames_per_second=20,
    log_progress=True
)

print(video_path)
