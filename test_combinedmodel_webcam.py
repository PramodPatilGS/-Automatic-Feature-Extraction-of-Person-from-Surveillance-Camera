from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

                    
# load model
model1 = load_model('inception_age.h5')
model2 = load_model('inception_gender.h5')
model3 = load_model('inception_spectacle.h5')
model4 = load_model('inception_beard2.h5')
# open webcam
webcam = cv2.VideoCapture(0)
    
classes1 = ['Age:Child', 'Age:Middle', 'Age:Old', 'Age:Young']
classes2 = ['Gender:Female', 'Gender:Male']
classes3 = ['Spectacle:YES', 'Spectacle:NO']
classes4 = ['Beard:NO', 'Beard:YES']
# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])
        face_crop1 = np.copy(frame[startY:endY+50, startX-30:startX+30])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        # preprocessing for face feature detection model
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop1 = cv2.resize(face_crop1, (224, 224))
        face_crop = face_crop.astype("float") / 255.0
        face_crop1 = face_crop1.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop1 = img_to_array(face_crop1)
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop1 = np.expand_dims(face_crop1, axis=0)
        # apply face feature detection on face
        conf1 = model1.predict(face_crop)[0]
        conf2 = model2.predict(face_crop)[0]
        conf3 = model3.predict(face_crop)[0]
        conf4 = model4.predict(face_crop1)[0]
        # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx1 = np.argmax(conf1)
        idx2 = np.argmax(conf2)
        idx3 = np.argmax(conf3)
        idx4 = np.argmax(conf4)
        label1 = classes1[idx1]
        label2 = classes2[idx2]
        label3 = classes3[idx3]
        label4 = classes4[idx4]
        label1 = "{}: {:.2f}%".format(label1, conf1[idx1] * 100)
        label2 = "{}: {:.2f}%".format(label2, conf2[idx2] * 100)
        label3 = "{}: {:.2f}%".format(label3, conf3[idx3] * 100)
        label4 = "{}: {:.2f}%".format(label4, conf4[idx4] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label1, (startX, Y-20),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, label2, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, label3, (startX, Y-40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, label4, (startX, Y-60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # display output
    cv2.imshow("Facial Attribute Detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

i = 0

while webcam.isOpened():

    # This condition prevents from infinte looping
    # incase video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Save Frame by Frame into disk using imwrite method
    # Save Frame by Frame into disk using imwrite method

    if i % 10 == 0:
        cv2.imwrite('Frame' + str(i) + '.jpg', frame)

    i += 1
# release resources
webcam.release()
cv2.destroyAllWindows()