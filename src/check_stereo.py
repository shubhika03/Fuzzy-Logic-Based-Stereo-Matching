import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skfuzzy as fuzz
from skfuzzy import control as ctrl 
from skimage.draw import circle


def find_disparity():
		
	#fuzzy logic
	d_to_left = ctrl.Antecedent(np.arange(0, 101, 1), 'left_edge')
	d_to_right = ctrl.Antecedent(np.arange(0, 101, 1), 'right_edge')
	sqe = ctrl.Antecedent(np.arange(0, 101, 1), 'sqe')
	perc = ctrl.Consequent(np.arange(0, 101, 1), 'perc')

	#defining rule base set for fuzzy logic
	d_to_left['low'] = fuzz.trimf(d_to_left.universe, [0, 0, 50])
	d_to_left['medium'] = fuzz.trimf(d_to_left.universe, [0, 50, 100])
	d_to_left['high'] = fuzz.trimf(d_to_left.universe, [50, 100, 100])
	d_to_right['low'] = fuzz.trimf(d_to_right.universe, [0, 0, 50])
	d_to_right['medium'] = fuzz.trimf(d_to_right.universe, [0, 50, 100])
	d_to_right['high'] = fuzz.trimf(d_to_right.universe, [50, 100, 100])	
	sqe['low'] = fuzz.trimf(sqe.universe, [0, 0, 50])
	sqe['medium'] = fuzz.trimf(sqe.universe, [0, 50, 100])
	sqe['high'] = fuzz.trimf(sqe.universe, [50, 100, 100])
	perc['low'] = fuzz.trimf(perc.universe, [0, 0, 50])
	perc['medium'] = fuzz.trimf(perc.universe, [0, 50, 100])
	perc['high'] = fuzz.trimf(perc.universe, [50, 100, 100])

	#fuzzy logic rule base
	
	rule1 = ctrl.Rule(sqe['low'] & d_to_left['low'], perc['high'])
	rule2 = ctrl.Rule(sqe['low'] & d_to_left['medium'], perc['high'])
	rule3 = ctrl.Rule(sqe['low'] & d_to_left['high'], perc['medium'])
	rule4 = ctrl.Rule(sqe['medium'] & d_to_left['low'], perc['medium'])
	rule5 = ctrl.Rule(sqe['medium'] & d_to_left['medium'], perc['medium'])
	rule6 = ctrl.Rule(sqe['medium'] & d_to_left['high'], perc['low'])
	rule7 = ctrl.Rule(sqe['high'] & d_to_left['low'], perc['medium'])
	rule8 = ctrl.Rule(sqe['high'] & d_to_left['medium'], perc['low'])
	rule9 = ctrl.Rule(sqe['high'] & d_to_left['high'], perc['low'])

	rule10 = ctrl.Rule(sqe['low'] & d_to_right['low'], perc['high'])
	rule11 = ctrl.Rule(sqe['low'] & d_to_right['medium'], perc['high'])
	rule12 = ctrl.Rule(sqe['low'] & d_to_right['high'], perc['medium'])
	rule13 = ctrl.Rule(sqe['medium'] & d_to_right['low'], perc['medium'])
	rule14 = ctrl.Rule(sqe['medium'] & d_to_right['medium'], perc['medium'])
	rule15 = ctrl.Rule(sqe['medium'] & d_to_right['high'], perc['low'])
	rule16 = ctrl.Rule(sqe['high'] & d_to_right['low'], perc['medium'])
	rule17 = ctrl.Rule(sqe['high'] & d_to_right['medium'], perc['low'])
	rule18 = ctrl.Rule(sqe['high'] & d_to_right['high'], perc['low'])

	perc_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
					rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18 ])
	percentage = ctrl.ControlSystemSimulation(perc_ctrl)
	
	
	point_x=[]
	point_y = []
	anss = []

	#generating 7 random points in left image and then find their corresponding position in right image
	for i in range(15):
		x = np.random.randint(100, 1000)
		y = np.random.randint(100, 1000)
		point_x.append(x)
		point_y.append(y)

	for l in range(15):
		val = -1
		ind = 0
		i = point_x[l]
		j = point_y[l]
		d1 = img_ll[i, j]
		d2 = img_lr[i, j]
		for k in range(j-130, j+30):
			
			if k >= 1368 or k <=22:
				continue
			else:
				ans = int(np.sum((img1[i-21:i+22,j-21:j+22,:] - img2[i-21:i+22, k-21:k+22,:])**2)*100/9)
				d_ll = np.abs(d1 - img_rl[i, k])*100/255
				d_rr = np.abs(d2 - img_rr[i, k])*100/255
				percentage.input['sqe'] = ans
				percentage.input['left_edge'] = d_ll
				percentage.input['right_edge'] = d_rr
				percentage.compute()
				z = percentage.output['perc']
				if z > val:
					val = z
					ind = k

		#print(val, ind)			
		anss.append(ind)			
	return point_x, point_y, anss
	
					
#Parameters
#img1: path of origina left image
#img2: path of original right image
#img_l: left gaussian blur image
#img_r: right gaussian blur image
#image_ll: distance of left nearest neighbour for left image
#image_lr: ditance of right nearest neighbour for left image
#image_rl: distance of left nearest neighbour for right image
#image_rr: ditance of right nearest neighbour for right image

img1 = mpimg.imread("view0.png")
img2 = mpimg.imread("view1.png")
img_l = mpimg.imread("img_l.png")
img_r = mpimg.imread("img_r.png")
img_ll = mpimg.imread("ll.png")
img_lr = mpimg.imread("lr.png")
img_rl = mpimg.imread("rl.png")
img_rr = mpimg.imread("rr.png")

rows, cols = img_l.shape
img_disp = np.zeros((rows, cols), np.uint8)

p1, p2, ans = find_disparity()

#for visualising the output
for i in range(len(p1)):
	rr1, cc1 = circle(p1[i], p2[i], 9)
	rr2, cc2 = circle(p1[i], ans[i], 9)
	col = list(np.random.choice(range(256), size=3)/255.0)
	img1[rr1, cc1] = col
	img2[rr2, cc2] = col

plt.imsave("img1.png", img1)
plt.imsave("img2.png", img2)	
