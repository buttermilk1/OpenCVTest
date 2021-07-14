import numpy as np
import cv2
import time
from past.builtins.misc import xrange
from logging import _startTime
import tkinter as tk



class State:
    def run(self):
        assert 0, "run not implemented"
    def next(self):
        assert 0, "next not implemented"
        
class StateMachine:
    def __init__(self,initState):
        self.currentState = initState
        self.currentState.run()
    def run(self):
        self.currentState.run()
        self.currentState = self.currentState.next()
        
class DetectObject(State):
    Capture = 0
    Boarder = 0
    
    def __init__(self, capture, boarder):
        self.Capture = capture
        self.Boarder = boarder
        
    def run(self):
        lower_green = np.array([30,60,60])
        upper_green = np.array([80,255,255])
        
        lower_blue = np.array([90,150,100])
        upper_blue = np.array([150,255,255])
        
        ret, frame = self.Capture.read()
    
        (tl,tr,br,bl) = self.Boarder
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        
        rect = np.zeros((4, 2), dtype = "float32")
        rect[0] = tl
        rect[1] = tr
        rect[2] = br
        rect[3] = bl
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        
        
        img = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        
        mask2 = cv2.inRange(img, lower_green, upper_green)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        mask3 = cv2.inRange(img, lower_blue, upper_blue)
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        
        cancel = False
        VehicleFront = [0,0]
        
        for i in xrange(0,img.shape[0]):
            if(cancel==True):
                break
            for j in xrange(0,img.shape[1]):
                if(mask2[i,j] == 255 ):
                    VehicleFront = [j, i]
                    cancel = True
                    break
            
        cancel = False
        VehicleBack = [0,0]
       
        for i in xrange(0,img.shape[0]):
            if(cancel==True):
                break
            for j in xrange(0,img.shape[1]):
                if(mask3[i,j] == 255 ):
                    VehicleBack = [j, i]
                    cancel = True
                    break
                
        cv2.polylines(frame, [np.array(self.Boarder)], True, (0,255,0), 3)
        cv2.drawMarker(warp, tuple(VehicleFront), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.drawMarker(warp, tuple(VehicleBack), color=(255,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
        
        cv2.imshow('warp', warp)  
        cv2.imshow('frame',frame)
    
        
    def next(self):
        return self
    

class Init(State):
    EdgePoints, InitComplete, Capture = 0,0,0
    
    def __init__(self, capture):
        self.EdgePoints = list()
        self.InitComplete = False
        self.Capture = capture
        
    def run(self):
        
        
        
        lower_red = np.array([0,130,130])
        upper_red = np.array([20,255,255])
        
        ret, frame = self.Capture.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        

    
        mask = cv2.inRange(img, lower_red, upper_red)

        self.EdgePoints = list()
        cancel = False
        EdgesFound = 0

        for i in xrange(0,frame.shape[0]//2):
            if(cancel==True):
                break
            for j in xrange(0,frame.shape[1]//2):
                if(mask[i,j] == 255):
                    self.EdgePoints.append([j, i])
                    cancel = True
                    EdgesFound+=1
                    break          
        cancel = False
               
        for i in xrange(0,frame.shape[0]//2):
            if(cancel==True):
                break
            for j in xrange(frame.shape[1]//2,frame.shape[1]):
                if(mask[i,j] == 255):
                    self.EdgePoints.append([j, i])
                    cancel = True
                    EdgesFound+=1
                    break
                
        cancel = False
              
        for i in xrange(frame.shape[0]//2,frame.shape[0]):
            if(cancel==True):
                break
            for j in xrange(frame.shape[1]//2, frame.shape[1]):
                if(mask[i,j] == 255):
                    self.EdgePoints.append([j, i])
                    cancel = True
                    EdgesFound+=1
                    break
        
        cancel = False      
    
        for i in xrange(frame.shape[0]//2,frame.shape[0]):
            if(cancel==True):
                break
            for j in xrange(0,frame.shape[1]//2):
                if(mask[i,j] == 255 ):
                    self.EdgePoints.append([j, i])
                    cancel = True
                    EdgesFound+=1
                    break
                
        if(EdgesFound >= 4):
            self.InitComplete = True

        cv2.polylines(frame, [np.array(self.EdgePoints)], True, (0,255,255), 3)
        cv2.imshow('frame',frame)
        #cv2.imshow('mask', mask)
                    
    def next(self):
        if(self.InitComplete==True): 
            return DetectObject(self.Capture, self.EdgePoints)
        return self
    
    
# HERE WE GO

#Setup Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT ,480)
ColorStateMachine = StateMachine(Init(cap))
window = tk.Tk()

def RunLoop():
    ColorStateMachine.run()
    time.sleep(0.1)

button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)

while(True):
    RunLoop()


#button.pack()
#window.after(100, RunLoop)
#window.mainloop()





# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()