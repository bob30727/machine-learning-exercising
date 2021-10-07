import os
import cv2
import numpy as np 
import csv

resolution_new_driver = 0.035  # 後來用穩定的大解析度新Driver掃描 (mm/pixel)
resolution_old_driver = 0.0844 # 舊的資料使用的小解析度(mm/pixel)
Y503_STANDARD_GRAY_DIFF = 170

path = "D:/code/1007_detect/"
#path = "D:/code/glorymakeup/dot_pattern_research/data28_FinalComparison/cut2"


def makeOutputDir(path, fullpath1):
	try:
		os.mkdir(os.path.join(path.replace("cut2","/output")))
	except:
		print("Output dir exists")

	try:
		os.mkdir(os.path.join(fullpath1.replace("cut2","/output")))
	except:
		print("Output dir exists")

	try:
		os.mkdir(os.path.join(path.replace("cut2","/grad")))
	except:
		print("Output dir exists")

	try:
		os.mkdir(os.path.join(fullpath1.replace("cut2","/grad")))
	except:
		print("Output dir exists")





def customizedOtsu(thImg):
	data = thImg.reshape(1, -1)
	data = data[data > 0]
	if not data.shape[0] % 2 == 0:
		data = np.delete(data,-1)

	data = data.reshape(2, int(data.shape[0]/2))

	#使用MatPlot繪出 histogram
	hist = cv2.calcHist([data], [0], None, [256], [0, 256])  # ←計算直方圖資訊
	# plt.figure()
	# plt.title("Grayscale Histogram")
	# plt.xlabel("Bins")
	# plt.ylabel("# of Pixels")
	# plt.plot(hist)
	# plt.xlim([0, 256])
	# plt.show();

	histData = hist.reshape(1, -1)
	print(np.argmax(histData))
	return np.argmax(histData)





def blendOutput(img, th):
	for i in range(th.shape[0]):
		for j in range(th.shape[1]):
			if th[i, j] > 0:
				img[i, j] = [255, 0, 0]

	return img


def blockCenterMask(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	roiImg = gray[int(img.shape[0]/2)-50:int(img.shape[0]/2)+50, int(img.shape[1]/2)-50:int(img.shape[1]/2)+50].copy()

	# 求roiImg裡面的極端值
	minValue = 255
	for i in range(roiImg.shape[0]):
		for j in range(roiImg.shape[1]):
			if roiImg[i, j] < 255:
				minValue = roiImg[i, j]

	ret, th = cv2.threshold(gray, minValue+30, 255, cv2.THRESH_BINARY)
	th = cv2.medianBlur(th, 5)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
	erode = cv2.erode(th,kernel,iterations = 2)

	return erode


files = os.listdir(path)
listX = [] # 放所有features(包含X, Y)
for f in files:
	fullpath1 = os.path.join(path, f)
	files2 = os.listdir(fullpath1)
	dotCount = 0
	feature = []
	makeOutputDir(path, fullpath1)
	for f2 in files2:
		dotCount += 1

		fullpath2 = os.path.join(fullpath1, f2)
		print(fullpath2)
		img = cv2.imread(fullpath2)
		#img = cv2.resize(img, (int(img.shape[0]/2), int(img.shape[1]/2)), interpolation=cv2.INTER_AREA)
		cv2.imshow("img", img)
		


		# 雜點影像開始計算
		input_img = img.copy()
		gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		#B, G, R = cv2.split(input_img)
		gray = cv2.GaussianBlur(gray, (3, 3), 0, borderType=cv2.BORDER_DEFAULT)

		cv2.imshow("gray", gray)



		grad_X = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		grad_Y = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		grad_X_64Fto8U=cv2.convertScaleAbs(grad_X)
		grad_Y_64Fto8U=cv2.convertScaleAbs(grad_Y)

		grad = cv2.addWeighted(grad_X_64Fto8U, 1.0, grad_Y_64Fto8U, 1.0, 0)
		grad = cv2.convertScaleAbs(grad)

		cv2.imwrite(os.path.join(fullpath1.replace("cut2","/grad"), (str(dotCount)+"_grad.jpg")), grad)


		# 改的地方
		maxValue = np.max(gray)
		minValue = np.min(gray)
		grayDiff = maxValue - minValue
		print(grayDiff)
		th_calibration = int(32 * grayDiff / Y503_STANDARD_GRAY_DIFF)
		#th_calibration = 32

		grad = cv2.GaussianBlur(grad, (3, 3), 0, borderType=cv2.BORDER_DEFAULT)
		ret, th = cv2.threshold(grad, th_calibration, 255, cv2.THRESH_TOZERO)
		customizedOtsu(th)

		th *= 4
		th_int = cv2.convertScaleAbs(th)


		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
		th_int = cv2.dilate(th_int,kernel,iterations = 1)

		mask = blockCenterMask(img)

		contourImg = img.copy()
		contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) > 8000 and cv2.contourArea(cnt) < 150000:
				cmax = []
				cmax.append(cnt)
				cv2.drawContours(th_int, cmax, -1, 0, thickness=-1)
				cv2.drawContours(contourImg, cmax, -1, (0,255,0), thickness=2)

		
		cv2.imshow("The noise will be blocked inside contour", contourImg)
		#outputImg = blendOutput(input_img, th_int)
		
		outputImg = input_img.copy()
		# 最後計算大雜點的顆數
		bigCount = 0
		ratio = float(resolution_new_driver/resolution_old_driver)

		contours2, hierarchy2 = cv2.findContours(th_int,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		hierarchy2 = np.squeeze(hierarchy2)
		for i in range(len(contours2)):
			x, y, w, h = cv2.boundingRect(contours2[i])
			if w+h > 15:
			#cv2.rectangle(input_img,(x,y),(x+w,y+h),(0,255,0),2)
				tmpShowImg = input_img[y:y+h, x:x+w]
				# 把很多白色區域的雜點踢掉
				binaryTmp = cv2.cvtColor(tmpShowImg, cv2.COLOR_BGR2GRAY)
				retTmp, thTmp = cv2.threshold(binaryTmp, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
				contoursTmp, hierarchyTmp = cv2.findContours(thTmp,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
				
				#改的地方
				if (len(contoursTmp)) > 1:
					continue
				if w/h > 1.5 or h/w > 1.5:
				   continue 


				cv2.imshow("thTmp", thTmp)
				#cv2.waitKey(0)
				bigCount += 1
				


			if w+h > 15 and w+h <= 20:
				cmax = []
				cmax.append(contours2[i])
				cv2.rectangle(outputImg,(x,y),(x+w,y+h),(0,0,255),2)
				

			if w+h > 20 and w+h <= 25:
				cmax = []
				cmax.append(contours2[i])
				cv2.rectangle(outputImg,(x,y),(x+w,y+h),(0,255,0),2)

			if w+h > 25 and w+h <= 30:
				cmax = []
				cmax.append(contours2[i])
				cv2.rectangle(outputImg,(x,y),(x+w,y+h),(255,0,0),2)


			if w+h > 30 and w+h <= 100:
				cmax = []
				cmax.append(contours2[i])

				cv2.rectangle(outputImg,(x,y),(x+w,y+h),(0,255,255),2)





		# 計算大雜點中，最大顆的大雜點直徑
		maxRadius = 0
		for i in range(len(contours2)):
			x, y, w, h = cv2.boundingRect(contours2[i])
			#cv2.rectangle(input_img,(x,y),(x+w,y+h),(0,255,0),2)
			if (w+h) > maxRadius:
				maxRadius = (w + h) 	

		cv2.imshow("grad", grad)
		cv2.imshow("th", th_int)
		cv2.imshow("output", outputImg)
		cv2.waitKey(1)
		cv2.imwrite(os.path.join(fullpath1.replace("cut2","/output"), (str(dotCount)+"_output.jpg")), outputImg)
		feature.append(bigCount)

	listX.append(feature)


xLabels = np.linspace(1, 4, 4)
X = np.array(listX)
print(X)
with open(os.path.join(path.replace("cut2",""), "file.csv"), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    for i in range(X.shape[0]):
    	writer.writerow(X[i])
    
