import cv2

class HumanFaceDetector():
    def __init__(self):

        # pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )

    # Implementation of face detection
    def detect_faces(self,img_path):
        '''
        INPUT:
        img_path    path of image

        OUTPUT:
        boolean     True if face detected

        Description:
        Returns ndarray with faces detected in image
        '''

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)

        return faces

