import cv2
import glob

try:
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
except AttributeError as e:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
for folder in glob.glob("data/*"):
    print(folder)
    vidfile = cv2.VideoWriter('{}/out-video.avi'.format(folder), fourcc,
                              30, (320,240))
                              
    for i,file in enumerate(glob.glob("{}/IMG/*center*.jpg".format(folder))):
        img = cv2.imread(file)
        vidfile.write(img)
        
        if i % 1000 == 0:
            print("\t{}".format(i))

    vidfile.release()