import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

def faceBox(faceNet, face_frame, detection):
    blob = cv2.dnn.blobFromImage(face_frame, 1.0, (300, 300), [104, 117, 123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []

    frameWidth = face_frame.shape[0]
    frameHeight = face_frame.shape[1]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            # cv2.rectangle(face_frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return face_frame, bboxs


#to set resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default = [1280, 720],
        nargs = 2,
        type = int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )

    while True:
        ret, frame = cap.read()
        
        result = model(frame)[0]

        detections = sv.Detections.from_yolov8(result)

        person_detections = []

        for detection in detections:
            _, confidence, class_id, _ = detection

            if model.model.names[class_id] == "person":
                person_detections.append(detection)

            labels = [f"{model.model.names[class_id]} {confidence: 0.2f}"
                for _, confidence, class_id, _ in person_detections]
            
            frame = box_annotator.annotate(
                scene = frame,
                detections = person_detections,
                labels = labels
            )

            face_frame, bboxs = faceBox(faceNet, frame, detection)

        for box in bboxs:
            face = face_frame[max(0,box[1]-padding):min(box[3]+padding,frame.shape[0]-1),max(0,box[0]-padding):min(box[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face_frame, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]


            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]

            age_gender_label="{},{}".format(gender,age)
            cv2.rectangle(frame,(box[0], box[1]-30), (box[2], box[1]), (0,255,0),-1) 
            cv2.putText(frame, age_gender_label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        
        person_count = len(person_detections)
        height, width, _ = frame.shape
        text_x, text_y = 10, 20
        cv2.putText(frame, f"People: {person_count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("yolov8 (Press 'esc' to close the window)", frame)
        # print(frame.shape)
        # break

        # if esc is pressed->stop loop
        if (cv2.waitKey(30) == 27): # 30ms wait & 27 - ASCII for escape
            break

if __name__ == "__main__":
    main()
