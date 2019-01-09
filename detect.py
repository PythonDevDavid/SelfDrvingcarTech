from __future__ import print_function
import numpy as np
import cv2

if __name__ == '__main__':
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade='])
    
    video_src = 'video.mp4'

    args = dict(args)
    cascade_fn = args.get('--cascade', "cascade_dir/cascade.xml")

    car_cascade = cv2.CascadeClassifier(cascade_fn)
    cap = cv2.VideoCapture(video_src)

    paused = False
    step = True

    while True:
        if not paused or step:
            flag, img = cap.read()

            height, width, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in cars:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 

            cv2.imshow('Cars', img)
        
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()