import cv2

image = cv2.imread('/Users/systemallica/Pictures/75398216_10214944708641736_3305056911426060288_o.jpg')
blur = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow("test", blur)
cv2.waitKey(0)


