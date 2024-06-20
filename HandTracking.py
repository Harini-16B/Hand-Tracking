import cv2 as cv
import mediapipe as mp
import time
ptime=0#previous time
cTime=0#current time

cap =cv.VideoCapture(0)
#initialise the module
mpHands =mp.solutions.hands
hands=mpHands.Hands()
mpDraw =mp.solutions.drawing_utils

while True:
    ret,frame=cap.read()
    imgRGB =cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results =hands.process(imgRGB)
    #checking multiple hands
    if results.multi_hand_landmarks:
        for lms in results.multi_hand_landmarks:
            for id,lm in enumerate(lms.landmark):
                #print(id,lm)
                h,w,c=frame.shape
                cx,cy =int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==20:
                    cv.circle(frame,(cx,cy),30,(255,0,0),cv.FILLED)

            mpDraw.draw_landmarks(frame,lms,mpHands.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2,circle_radius=2))
    cTime=time.time()
    fps=1/(cTime-ptime)
    ptime=cTime

    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)



    cv.imshow("Image",frame)#to run the frame
    if  cv.waitKey(1) &0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
