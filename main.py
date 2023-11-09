# import contours as contours
# from imageai.Detection import ObjectDetection
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

# import cv2
# import pytesseract
#
# image = cv2.imread('1.jpg')
# height, widht, _ = image.shape
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cnts, _ = contours.sort_countorus(cnts[0])
# cnts, _ = contours
#
# for c in cnts:
#     area = cv2.contourArea(c)
#     x, y, w, h = cv2.boundingRect(c)
#     if area > 5000:
#         img = image[y: y+h, x: x+w]
#         result = pytesseract.image_to_string(img, lang='rus+eng')
#         print(result)
# cv2.imshow('test', img)
# cv2.waitKey()

# import matplotlib
# import matplotlib.pyplot as plt
# import pytesseract
# import cv2
#
# matplotlib.use('Agg')
#
#
# def open_img(img_path):
#     carplate_img = cv2.imread(img_path)
#     carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
#     plt.axis('off')
#     plt.imshow(carplate_img)
#     return carplate_img
#
#
# def carplate_extract(image, carplate_haar_cascade):
#     carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
#
#     for x, y, w, h in carplate_rects:
#         # carplate_img = image[y-5:y+h+10, x+5:x+w-10]
#         carplate_img = image[y+15:y+h-10, x+15:x+w-20]
#
#     return carplate_img
#
#
# def enlarge_img(image, scale_percent):
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     plt.axis('off')
#     resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     return resized_image
#
#
# def main():
#     carplate_img_rgb = open_img(img_path='3.jpg')
#     carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
#
#     carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
#     carplate_extract_img = enlarge_img(carplate_extract_img, 150)
#
#     carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
#
#     print('Номер авто: ', pytesseract.image_to_string(
#         carplate_extract_img_gray,
#         config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
#           )
#     cv2.imshow('test', carplate_extract_img_gray)
#     cv2.waitKey()
#
# if __name__ == '__main__':
#     main()


from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = '3.mp4'
cap = cv2.VideoCapture(video_path)
ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)
        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()
        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
