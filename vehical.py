import cv2
from cv2 import dilate
import numpy as np

#Caputer the video or open webCamera
#cap = cv2.VideoCapture('Car_Video.avi')
cap = cv2.VideoCapture('video.mp4')

count_line_pos =550
# Initialize Substurctor 
# this detect the vehical not the other things will detect from background
algo = cv2.createBackgroundSubtractorMOG2()



min_width_rect = 80 #min_width of rectangle
min_height_rect = 80 #min_heigth of reactangle

def center_handle(x,y,w,h):
    x1 = int(w/2) 
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

def center_handle_color(x,y,w,h):
    x1 = int(w/2) 
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    hsv_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
    # heigh,width, _ = frame1.shape

    pixel_pos = hsv_frame[cy,cx]
    # Cy = int(heigh / 2)
    # Cx = int(width / 2)
    # print(pixel_pos)
    hue_value = pixel_pos[0]
    color = "undefined"
    if hue_value < 30:
        color = "Dark Black"
    elif hue_value < 70:
        color = "black"
    elif hue_value < 56:
        color = "Green"
    elif hue_value < 120:
        color = "White"
    elif hue_value < 190:
        color = "Silver"
    else:
        color = "No Detect"

    pixel_bgr = frame1[cy,cx]
    b ,g ,r = int(pixel_bgr[0]) , int(pixel_bgr[1]), int(pixel_bgr[2])
    cv2.putText(frame1,color,(10,70),0,1.5,(0,0,0),2)
    
    return color
# after decteing will store on list 
detect = []
offset = 6 # allow error between pixel 
couter = 0


while True:
    #read the video
    ret, frame1 = cap.read()
    # frame convert to gray frame 
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # this will point the vehical
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # appyling on each frame and alogrithm will apply on them
    img_sub = algo.apply(blur)
    # the function dilates the source image using the specified structuring
    #  element that determines the shape of a pixel neighborhood over which
    #  the maximam is taken , np for array , return given shape
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    #this will make one perfect structur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # this handle the multichannel image and gives the shape of vehical
    dilatad =  cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatad = cv2.morphologyEx(dilatad,cv2.MORPH_CLOSE,kernel) 
    # help to find to object on frame
    counterShape,h = cv2.findContours(dilatad,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Gray img showing
    #cv2.imshow('Detector',dilatad)
    #               start the line      end the line          colourline  thickness
    cv2.line(frame1,(25,count_line_pos),(1200,count_line_pos),(255,127,0),3)
    # this for rectangle draw on car
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        
        validate_counter = (w>= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
#       this will draw the rectangle on each car and randomaly generate
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        # this will detect the colour of car and return the colour name
        car_colour = center_handle_color(x,y,w,h)

        center = center_handle(x,y,w,h)
        # this draw the point on center rectangle
        cv2.circle(frame1,center,4,(0,255,0),-1)
        # after detecting car ,store on list[]
        detect.append(center)


        for (x,y) in detect:
            # after the line cross count the car 
            if y < (count_line_pos+offset) and y > (count_line_pos-offset):
             couter+=1
            #  after cross the line vehical, line colour will change    
             cv2.line(frame1,(25,count_line_pos),(1200,count_line_pos),(0,127,255),3)
            #  remove one then again detect
             detect.remove((x,y))

            #  print("vehical Count"+ str(couter))
            # this will print 
             csvFileToWrite = open("vehicalcounting.csv", "a")
             csvFileToWrite.write("\n vehical Counting No : "+str(couter))
             csvFileToWrite.write("\n vehical Colour  : "+str(car_colour))
             csvFileToWrite.close()

    #  puttext will put text on Video frame
    cv2.putText(frame1,"Vehical counter : "+str(couter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),5)
    # Show the video 
    cv2.imshow('Vehical Detection System',frame1)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()