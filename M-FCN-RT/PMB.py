
import numpy as np
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from PIL import Image
from skimage.measure import label
import os
import random
from skimage import color
from skimage.morphology import extrema
import cv2
from PIL import Image
import PIL.ImageOps
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
rand_cmap = ListedColormap(np.random.rand(256,3))
img = cv2.imread('C:/Users/123/Desktop/lun/101.png')
img3 = cv2.imread('C:/Users/123/Desktop/lun/103.png')
img2 = cv2.imread('C:/Users/123/Desktop/lun/100.png')
img1=img

from matplotlib.colors import ListedColormap
rand_cmap = ListedColormap(np.random.rand(256,3))


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1 = gray 


ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_OTSU)
# newimg = cv2.bitwise_not(thresh)
newimg = thresh
# cv2.imshow('p',newimg)
# cv2.waitKey(0)
plt.imshow(newimg,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(newimg,cv2.MORPH_OPEN,kernel, iterations = 1)
# cv2.imshow('p',opening)
# cv2.waitKey(0)
plt.imshow(opening,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)#
# Finding sure foreground area
# cv2.imshow('p',sure_bg)
# cv2.waitKey(0)
plt.imshow(sure_bg,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# dist_transform = cv2.distanceTransform(opening,1,5)
dist_transform = ndimage.distance_transform_edt(opening)
# minima = extrema.h_minima(-dist_transform,1)
# cv2.imshow('d',dist_transform)
# cv2.waitKey(0)
plt.imshow(dist_transform)
plt.axis('off')
plt.show()
# plt.imshow(minima) 
# plt.axis('off')
# plt.show()
# overlay_h = color.label2rgb(minima, img, alpha=0.7, bg_label=0,
#                             bg_color=None, colors=[(1, 0, 0)])

# plt.imshow(overlay_h)
# plt.axis('off')
# plt.show()    

ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)                       
# ret, sure_fg = cv2.threshold(dist_transform,minima.min(),255,0)#
# localMax = peak_local_max(dist_transform, indices=False, min_distance=10,
# 	# footprint=np.ones((3, 3)),
# 	labels=opening)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
# cv2.imshow('p',sure_fg)
# cv2.waitKey(0)
plt.imshow(sure_fg,cmap=plt.cm.gray)
plt.axis('off')
plt.show()


gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,10,100,cv2.THRESH_BINARY)
print(thresh.shape)

gray = sure_fg
ret, thresh1 = cv2.threshold(gray,128,155,cv2.THRESH_BINARY)
print(thresh1.shape)

# plt.imshow(thresh,cmap='gray')
# plt.axis('off')
# plt.savefig('C:/Users/123/Desktop/lun/sprit11.jpg', format='jpg', transparent=True, dpi=1000, pad_inches = 0)
# plt.show()
# plt.imshow(thresh1,cmap='gray')
# plt.axis('off')
# plt.savefig('C:/Users/123/Desktop/lun/sprit12.jpg', format='jpg', transparent=True, dpi=1000, pad_inches = 0)
# plt.show()
cv2.imshow("result", thresh)
 
cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.imshow("result", thresh1)
 
cv2.waitKey(0) 
cv2.destroyAllWindows()
img4= cv2.add(thresh, thresh1)

plt.imshow(img4,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# cv2.imshow("result", img4)
 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
print(img4.shape)
rows,cols=img4.shape
print()
for i in range(rows):
  for j in range(cols):
    if (img4[i,j]==255):
      img4[i,j]=0
    elif (img4[i,j]==155):
      img4[i,j]=255
    elif (img4[i,j]==100):
      img4[i,j]=0
    else:
      img4[i,j]=0
print(img4)
plt.figure("lena")
plt.imshow(img4,cmap=plt.cm.gray)
plt.axis('off')
plt.show()


# kernel = np.ones((1,1),np.uint8)
# img4 = cv2.morphologyEx(img4,cv2.MORPH_OPEN,kernel, iterations = 1)
# plt.figure("lena")
# plt.imshow(img4,cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()

# kernel = np.ones((4,4),np.uint8)
# img4 = cv2.morphologyEx(img4,cv2.MORPH_CLOSE,kernel, iterations = 1)
# plt.figure("lena")
# plt.imshow(img4,cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()


unknown = cv2.subtract(sure_bg,img4)#
cv2.imshow('p',unknown)
cv2.waitKey(0)

ret, markers = cv2.connectedComponents(img4)
markers1=markers
print(markers1.shape)
cv2.imshow('p',markers1)
cv2.waitKey(0)
markers = markers
markers[unknown==255]=0
print(markers.shape)

#Selectable
# kernel = np.ones((1,1),np.uint8)
# su = cv2.dilate(markers,kernel,iterations=3)#
# plt.imshow(su)
# plt.show()
# markers = cv2.watershed(img,markers)
# # img2[markers == -1] = [255,0,0]
# # plt.imshow(img2)
# # plt.show()
# # plt.imshow(markers)
# # plt.show()
# img2[markers == -1] = [255,0,0]
# plt.imshow(img2)
# plt.axis('off')
# fig = plt.gcf()  
# fig.set_size_inches(80.0/10,80.0/10)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())  
# plt.gca().yaxis.set_major_locator(plt.NullLocator())  
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
# plt.margins(0,0)  
# fig.savefig('C:/Users/123/Desktop/lun/sprite2.jpg', format='jpg', transparent=True, dpi=1000, pad_inches = 0)
# fig.savefig('C:/Users/123/Desktop/lun/sprite3.pdf', format='pdf', transparent=True, dpi=1000, pad_inches = 0)  
# plt.show()


labels = watershed(-dist_transform, markers,mask=opening)
# labels = cv2.watershed(img,markers)
basins = np.ma.masked_array(labels, mask=(labels <= 0))
cv2.imshow('p',markers)
cv2.waitKey(0)
# cv2.imshow('p',img)
# cv2.waitKey(0)
# mask=opening

for label in np.unique(markers):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
 
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[markers == label] = 255
 
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
# 	for c in cnts: 
# # compute the center of the contour 
# 		M = cv2.moments(c) 
# 		cX = int(M["m10"] / M["m00"]) 
# 		cY = int(M["m01"] / M["m00"])# draw the contour and center of the shape on the image 
# 		# cv2.drawContours(img, [c], -1, (0, 255, 0), 2) 
# 		# cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1) 
# 		cv2.putText(img, "+", (cX -10, cY ), 
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)# show the image 
	# cv2.imshow("Image", img) 
	# cv2.waitKey(0)
	ce = max(cnts, key=cv2.contourArea)
 
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(ce)
	# cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(img2, "+", (int(x)-10, int(y)+10),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
b,g,r = cv2.split(img2)    
img2 = cv2.merge([r,g,b])    

plt.imshow(img2)
plt.axis('off')
fig = plt.gcf()  
fig.set_size_inches(80.0/10,80.0/10)
plt.gca().xaxis.set_major_locator(plt.NullLocator())  
plt.gca().yaxis.set_major_locator(plt.NullLocator())  
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
plt.margins(0,0)  
fig.savefig('C:/Users/123/Desktop/lun/sprite.jpg', format='jpg', transparent=True, dpi=1000, pad_inches = 0)
fig.savefig('C:/Users/123/Desktop/lun/sprite1.pdf', format='pdf', transparent=True, dpi=1000, pad_inches = 0)  
plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("gray")
ax1.imshow(-dist_transform, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title("Distance")
ax2.imshow(markers1, cmap=rand_cmap, interpolation='nearest')
ax2.set_title("Markers")

ax3.imshow(labels , cmap=rand_cmap, interpolation='nearest')
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
mark = np.unique(markers)
print(markers.shape)
rows,cols=basins.shape

gray = np.zeros((rows, cols), dtype=np.uint8)
gray = np.array(gray)
plt.imshow(gray, cmap=plt.cm.gray)
plt.imshow(basins ,vmin=0, cmap=rand_cmap, interpolation='none')
plt.axis('off')
fig = plt.gcf()  
fig.set_size_inches(80.0/10,80.0/10)
plt.gca().xaxis.set_major_locator(plt.NullLocator())  
plt.gca().yaxis.set_major_locator(plt.NullLocator())  
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
plt.margins(0,0)  
fig.savefig('C:/Users/123/Desktop/lun/sprit.jpg', format='jpg', transparent=True, dpi=1000, pad_inches = 0)
fig.savefig('C:/Users/123/Desktop/lun/sprit1.pdf', format='pdf', transparent=True, dpi=1000, pad_inches = 0)
plt.show()
