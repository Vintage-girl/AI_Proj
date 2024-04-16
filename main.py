# import cv2


# def facebox(faceNet,frame):
#     # print(frame)
#     frameHeight= frame.shape[0]
#     frameWidth= frame.shape[1]
#     blob= cv2.dnn.blobFromImage(frame,1.0,(300,300),[104,117,123],swapRB=False)
#     faceNet.setInput(blob)
#     detection =faceNet.forward()
#     # print(detection.shape)
#     bboxs =[]
#     for i in range(detection.shape[2]):
#         confidence = detection[0,0,i,2]
#         if confidence>0.7:
#             x1 = int(detection[0,0,i,3]*frameWidth)
#             y1 = int(detection[0,0,i,4]*frameHeight)
#             x2 = int(detection[0,0,i,5]*frameWidth)
#             y2 = int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
#     # return detection
#     return frame, bboxs

# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"


# faceNet = cv2.dnn.readNet(faceModel,faceProto)

# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# video = cv2.VideoCapture(0)

# padding = 20

# while True:
#     ret,frame = video.read()
#     # detect = facebox(faceNet, frame)
#     frame,bboxs = facebox(faceNet, frame)
#     for bbox in bboxs:
#         # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

#         genderNet.setInput(blob)
#         genderPred=genderNet.forward()
#         gender=genderList[genderPred[0].argmax()]

#         ageNet.setInput(blob)
#         agePred=ageNet.forward()
        
#         age_index = agePred[0].argmax()

#         # Get the corresponding age category
#         age_category = ageList[age_index]

#         # Extract the lower and upper bounds of the age category
#         age_lower_bound = int(age_category.split('(')[1].split('-')[0])
#         age_upper_bound = int(age_category.split('-')[1].split(')')[0])

#         # Calculate the average age
#         predicted_age = (age_lower_bound + age_upper_bound) // 2

#         label="{},{}".format(gender,predicted_age)

#         if predicted_age < 18:
#            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,0,255),-1)
#            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#            cv2.imshow("Capture",frame)
#            k = cv2.waitKey(1)
#            if k == ord('q') or 32 :
#                break
#         #    video.release()
#         #    cv2.destroyAllWindows
#         else:
#             cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1)
#             cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#             cv2.imshow("Capture",frame)
#             k = cv2.waitKey(1)
#             # if k == ord('q') or 32 :
#             #     break
#             # video.release()
#             # cv2.destroyAllWindows

#         # cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), box_color,-1) 

#         # cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)

#     cv2.imshow("Capture",frame)
#     k = cv2.waitKey(1)
#     if k == ord('q') or 32 :
#         break
# video.release()
# cv2.destroyAllWindows





# def faceBox(faceNet,frame):
#     frameHeight=frame.shape[0]
#     frameWidth=frame.shape[1]
#     blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
#     faceNet.setInput(blob)
#     detection=faceNet.forward()
#     bboxs=[]
#     for i in range(detection.shape[2]):
#         confidence=detection[0,0,i,2]
#         if confidence>0.7:
#             x1=int(detection[0,0,i,3]*frameWidth)
#             y1=int(detection[0,0,i,4]*frameHeight)
#             x2=int(detection[0,0,i,5]*frameWidth)
#             y2=int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
#     return frame, bboxs


# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"



# faceNet=cv2.dnn.readNet(faceModel, faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']


# video=cv2.VideoCapture('4.mp4')

# padding=20

# while True:
#     ret,frame=video.read()
#     frame,bboxs=faceBox(faceNet,frame)
#     for bbox in bboxs:
#         # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPred=genderNet.forward()
#         gender=genderList[genderPred[0].argmax()]


#         ageNet.setInput(blob)
#         agePred=ageNet.forward()
#         age=ageList[agePred[0].argmax()]


#         label="{},{}".format(gender,age)
#         cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
#         cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#     cv2.imshow("Age-Gender",frame)
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()


import cv2


def facebox(faceNet,frame):
    # print(frame)
    frameHeight= frame.shape[0]
    frameWidth= frame.shape[1]
    blob= cv2.dnn.blobFromImage(frame,1.0,(300,300),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection =faceNet.forward()
    # print(detection.shape)
    bboxs =[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
    # return detection
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


faceNet = cv2.dnn.readNet(faceModel,faceProto)

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

padding = 20


# while True:
#     ret, frame = video.read()
#     frame, bboxs = facebox(faceNet, frame)
#     for bbox in bboxs:
#         face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#                max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

#         genderNet.setInput(blob)
#         genderPred = genderNet.forward()
#         gender = genderList[genderPred[0].argmax()]

#         ageNet.setInput(blob)
#         agePred = ageNet.forward()

#         # Get the predicted age category index
#         age_index = agePred[0].argmax()

#         # Get the corresponding age category
#         age_category = ageList[age_index]

#         # Extract the lower and upper bounds of the age category
#         age_lower_bound = int(age_category.split('(')[1].split('-')[0])
#         age_upper_bound = int(age_category.split('-')[1].split(')')[0])

#         # Calculate the average age
#         predicted_age = (age_lower_bound + age_upper_bound) // 2

#         label = "{},{}".format(gender, predicted_age)

#         # Check if predicted age is below 18
#         if predicted_age < 18:
#             box_color = (0, 0, 255)  # Red color for warning
#         else:
#             box_color = (0, 255, 0)  # Green color for normal

#         cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), box_color, -1)
#         cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
#                     cv2.LINE_AA)

#     cv2.imshow("Capture", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break  # Break the loop only if 'q' is pressed

# # Release the video capture object and close all windows
# video.release()
# cv2.destroyAllWindows()


while True:
    ret, frame = video.read()
    frame, bboxs = facebox(faceNet, frame)
    for bbox in bboxs:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()

        # Get the predicted age category index
        age_index = agePred[0].argmax()

        # Get the corresponding age category (range)
        age_category = ageList[age_index]

        label = "{},{}".format(gender, age_category)

        # Check if predicted age is below 18
        if age_category.startswith('(0') or age_category.startswith('(4') or age_category.startswith('(8') or age_category.startswith('(15'):
            box_color = (0, 0, 255)  # Red color for warning
        else:
            box_color = (0, 255, 0)  # Green color for normal

        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), box_color, -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow("Capture", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break  # Break the loop only if 'q' is pressed

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()


