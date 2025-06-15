import cv2 as cv
import numpy as np
import os 


imgs = [image for image in os.listdir('data/') if image.endswith(('.jpg'))]
if len(imgs) == 0:
    print('no images found')
else:
    print('images are:', imgs)

for image in imgs:
    img_path = os.path.join('data/', image)
    img = cv.imread(img_path)

    img = cv.resize(img, (500,500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

    if len(faces_rect) >= 1:
        for (x, y, w, h) in faces_rect:
            cv.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv.putText(gray, 'face detected', (x-15, y-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv.putText(gray, 'no face detected', (x-15, y-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    print(f'number of faces found in {image}= {len(faces_rect)}')

    cv.imshow(f'detected: {image}', gray)
    cv.imwrite(f'results/{image[0]}.jpg', gray)


cv.waitKey(0)
cv.destroyAllWindows()

