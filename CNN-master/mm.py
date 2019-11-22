from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
# img = cv2.imread('q.jpg', cv2.IMREAD_UNCHANGED)
# # height, width, dim = img.shape
# height = img.shape[0]
# width = img.shape[1]
# dim = img.shape[2]
# print("height",height+"width",width+"dim",dim)
# img = img[(height/4):(3*height/4), (width/4):(3*width/4), :]
# height = img.shape[0]
# width = img.shape[0]
# dim = img.shape[2]
#
# img_vec = np.reshape(img, [height * width, dim] )
#
# kmeans = KMeans(n_clusters=3)
# kmeans.fit( img_vec )
#
# unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
# sort_ix = np.argsort(counts_l)
# sort_ix = sort_ix[::-1]
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x_from = 0.05
#
# for cluster_center in kmeans.cluster_centers_[sort_ix]:
#     ax.add_patch(patches.Rectangle( (x_from, 0.05), 0.29, 0.9, alpha=None,
#                                     facecolor='#%02x%02x%02x' % (cluster_center[2], cluster_center[1], cluster_center[0] ) ) )
#     x_from = x_from + 0.31
#
# plt.show()

# import cv2

# read image
img = cv2.imread('q.jpg', cv2.IMREAD_UNCHANGED)

# get dimensions of image
dim = img.shape

# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
print('Image Dimension    : ', dim)
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Number of Channels : ', channels)

img = img[125:375, 125:375]

img_vec = np.reshape(img, [250000, dim] )

kmeans = KMeans(n_clusters=3)
kmeans.fit( img_vec )

unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
sort_ix = np.argsort(counts_l)
sort_ix = sort_ix[::-1]

fig = plt.figure()
ax = fig.add_subplot(111)
x_from = 0.05

for cluster_center in kmeans.cluster_centers_[sort_ix]:
    ax.add_patch(patches.Rectangle( (x_from, 0.05), 0.29, 0.9, alpha=None,
                                    facecolor='#%02x%02x%02x' % (cluster_center[2], cluster_center[1], cluster_center[0] ) ) )
    x_from = x_from + 0.31

plt.show()