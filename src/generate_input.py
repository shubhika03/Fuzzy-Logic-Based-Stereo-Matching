import cv2
import numpy as np
import argparse

def nearest_neighbour():
	#finds the neares distnace of left and right neighbour usinf sobel derivative
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
	sobelx = np.absolute(sobelx)
	sobel_8u = np.uint8(sobelx)

	rows, cols = img.shape


	border_threshold = 23 #40 #32
	ret, temp = cv2.threshold(sobel_8u, border_threshold, 255, cv2.THRESH_BINARY)
	kernel = np.ones((3, 3), np.uint8)
	temp = cv2.erode(temp, kernel, 2)
		
	left = np.zeros((rows, cols, 1), np.uint8)
	right = np.zeros((rows, cols, 1), np.uint8)

	for i in range(0, rows):
		for j in range(0, cols):
			if j == 0:
				left[i, j] = 0
			elif temp[i, j] == 255:
				left[i, j] = 0
			else:
				ans = left[i, j-1] + 1
				if ans <= 255:
					left[i, j] = ans
				else:
					left[i, j] = 255

	for i in range(0, rows):
		for j in range(cols-1, -1, -1):
			if j == cols-1:
				right[i, j] = 0
			elif temp[i, j] == 255:
				right[i, j] = 0
			else:
				ans = right[i, j+1] + 1
				if ans <= 255:
					right[i, j] = ans
				else:
					right[i, j] = 255				

		
	left = cv2.medianBlur(left, 7)		
	right = cv2.medianBlur(right, 7)					
	#returns the left and right image having distance of nearest left and right pixel stored at each pixel
	return left, right

parser = argparse.ArgumentParser('Arguments for code')
parser.add_argument('-left', action='store_true',help='true for left image')
parser.add_argument('-right', action='store_true',help='true for righ image')
parser.add_argument("-path",default=" ",help="path of image")
args = parser.parse_args()

img = cv2.imread(args.path, 0)
img = cv2.GaussianBlur(img,(7,7),0)
ans1, ans2 = nearest_neighbour()

#uncomment for visualisation of the data generated
#cv2.imshow("image", img)
#cv2.imshow("right_nearest", ans2)
#cv2.imshow("left nearest", ans1)
#cv2.waitKey(0)

#writes the input data generated
#img: gaussian blur image
#image_ll: distance of left nearest neighbour for left image
#image_lr: ditance of right nearest neighbour for left image
#image_rl: distance of left nearest neighbour for right image
#image_rr: ditance of right nearest neighbour for right image

if args.left:
	cv2.imwrite("img_l.png", img)
	cv2.imwrite("lr.png", ans2)
	cv2.imwrite("ll.png", ans1)
if args.right:
	cv2.imwrite("img_r.png", img)
	cv2.imwrite("rr.png", ans2)
	cv2.imwrite("rl.png", ans1)
