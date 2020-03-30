import cv2 as cv
import time

cap = cv.VideoCapture(0)

text = input('Нажми на Enter что бы сфоткаться:')

while (True):
	ret, frame = cap.read()
	if text == "":
		print('[INFO]Фото грузиться')
		cv.imwrite('photo.jpg', frame)
		break

cap.release()
cv.destroyAllWindows()
