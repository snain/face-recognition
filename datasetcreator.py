import cv2
import sqlite3
def insert(Id,name):
    conn=sqlite3.connect('facebase.db')
    print("pass")  
    cmd="SELECT  * FROM person WHERE Id="+str(Id)
    cur=conn.execute(cmd)
    isrecord=0
    for row in cur:
        isrecord==1    
    if isrecord==1:
        cmd="UPDATE person SET name='"+str(name)+"' WHERE Id="+str(Id)
    else:
        cmd="INSERT INTO person(Id,name) Values('"+str(Id)+"','"+str(name)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()
        
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Id=int(input('enter your id: '))
#name=input('enter name')
#insert(Id,name)
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>25:
        break
cam.release()
cv2.destroyAllWindows()