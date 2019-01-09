from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt


#Which video clip to analyze
insert = input("Enter the name+format of the video to anyalyze: ")

#Video = The insert Clip
video = cv2.VideoCapture(insert)

#So program know that color of the lines
w_line = input("What color of lines: ")

if __name__ == '__main__':
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade='])
    
    video_src = insert

    args = dict(args)
    cascade_fn = args.get('--cascade', "cascade_dir/cascade.xml")

    car_cascade = cv2.CascadeClassifier(cascade_fn)
    cap_c = cv2.VideoCapture(video_src)

    paused = False
    step = True

while True:
    #Take the video input and make it frame by frame
    ret, orgi_frame = video.read()
    if not ret:
        video = cv2.VideoCapture(insert)

    #Crop the Edage image
    inputV = orgi_frame[500:720, 50:1240]

    #chaning color space and adding a GaussianBlur
    frame1 = cv2.GaussianBlur(orgi_frame,(5,5),0)
    frame = cv2.GaussianBlur(inputV, (5,5),0)
    #Makes Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray ,50,250)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Shows what a yellow line is
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges_y = cv2.Canny(mask, 75, 150)

#if Lines is white
    if w_line == "white":
        print("White")
        #Generates the Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=80)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(frame,(x1, y1),(x2, y2),(0,0, 255), 3)
#If lines is yellow
    else:
        print("Yellow")
        lines_y = cv2.HoughLinesP(edges_y, 1, np.pi/180, 50, maxLineGap=50)
        if lines_y is not None:
            for line_y in lines_y:
                x1, y1, x2, y2 = line_y[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 9)

    if not paused or step:
            flag, img = cap_c.read()

            height, width, c = img.shape
            gray_c = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray_c, 1.2, 5)

            for (x,y,w,h) in cars:
                cv2.rectangle(orgi_frame,(x,y),(x+w,y+h),(0,0,255),2) 

        


    #Resizeing
    gray = cv2.resize(gray, (0, 0), None, .50, .50 )
    edges = cv2.resize(edges, (0, 0), None, .50, .50)
    frame = cv2.resize(frame, (0,0), None, .50, .50)
    orgi_frame = cv2.resize(orgi_frame,(0,0), None, .50, .50)
    hsv = cv2.resize(hsv,(0,0), None, .50, .50)

    #edgr = edages + gray + hsv
    edgr = np.hstack((edges, gray))


    #Resixing edgr
    edgr = cv2.resize(edgr, (0,0), None, .80, .80)

    #Shows the Windows
    cv2.imshow("OverView", edgr)
    cv2.imshow("Frame", frame)
    cv2.imshow("Input", orgi_frame)
    cv2.imshow("Hsv color space", hsv)

#Makes sure that the program plays
    key = cv2.waitKey(25)
    if key == 27:
        break

#When the program exits
video.release()
cv2.destroyAllWindows()
