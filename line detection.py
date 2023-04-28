import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import pyttsx3 
from gtts import gTTS

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

img=cv2.imread("book.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

h,w,c = img.shape
if w > 100:
    new_w=1000
    ar=w/h
    new_h=int(new_w/ar)
    img=cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_AREA)

plt.imshow(img)

def threshoding(image):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thres=cv2.threshold(img_gray,80,225,cv2.THRESH_BINARY_INV)
    plt.imshow(thres,cmap="gray")
    return thres

thres_img=threshoding(img)   
kernal=np.ones((3,85),np.uint8) 
dilated=cv2.dilate(thres_img,kernal,iterations=1)
plt.imshow(dilated,cmap="gray")


(contours,heirarchy)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#(x,y,w,h)
sorted_contours_lines=sorted(contours,key=lambda ctr:cv2.boundingRect(ctr)[1])

img2=img.copy()
for ctr in sorted_contours_lines:
    x,y,w,h =cv2.boundingRect(ctr)
    cv2.rectangle(img2,(x,y),(x+w,y+h),(40,100,250),2)
    
plt.imshow(img2)




