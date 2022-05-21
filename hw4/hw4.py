import cv2
import numpy as np
import math

sample1 = cv2.imread("hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
sample2 = cv2.imread("hw4_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
sample3 = cv2.imread("hw4_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)

# p1 (a)
I2 = np.array([[1, 2], [3, 0]], dtype = 'uint8')
N = len(I2)
threshold = 255 * (I2 + 0.5) / (N * N)

h, w = sample1.shape
threshold = np.tile(threshold, (h//N, w//N))

result1 = np.zeros((h,w))
result1[sample1 >= threshold] = 1

#p1 (b)
n = 2
I = np.array([[1, 2], [3, 0]], dtype = 'uint8')

for i in range(7):
    I2 = np.zeros((n*2, n*2))

    I2[0:n, 0:n] = I*4 + 1
    I2[0:n, n: ] = I*4 + 2
    I2[n: , 0:n] = I*4 + 3
    I2[n: , n: ] = I*4 + 0
    
    I = I2
    n*=2

I256 = I

N = len(I256)
threshold = 255 * (I256 + 0.5) / (N * N)

h, w = sample1.shape
result2 = np.zeros((h,w))
result2[sample1 >= threshold] = 1

#p1 (c)

# Floyd-Steinberg
result3 = np.copy(np.lib.pad(sample1,(1,1),'constant')) / 255

kernel = [[0, 0, 7/16],
        [3/16, 5/16, 1/16]]

ones = np.ones((2,3))

Height, Width = result3.shape

for y in range(1, Height-1):
    for x in range(1, Width-1):

        old_value = result3[y, x]
        new_value = 0
        if (old_value >= 0.5) :
            new_value = 1

        Error = old_value - new_value
        
        patch = result3[y:y+2, x-1:x+2]
        
        NewNumber = patch + Error * ones * kernel
        NewNumber[NewNumber>1] = 1
        NewNumber[NewNumber<0] = 0
        
        result3[y:y+2, x-1:x+2] = NewNumber
        result3[y, x] = new_value
            
result3 = result3[1:257, 1:257]

# Jarvis’ 
result4 = np.copy(np.lib.pad(sample1,(2,2),'constant')) / 255

kernel = np.array([[0, 0, 0, 7, 5],
         [3,5,7,5,3],
         [1,3,5,3,1]])/48

ones = np.ones((3,5))

Height, Width = result4.shape

for y in range(2, Height-2):
    for x in range(2, Width-2):

        old_value = result4[y, x]
        new_value = 0
        if (old_value >= 0.5) :
            new_value = 1

        Error = old_value - new_value
        
        patch = result4[y:y+3, x-2:x+3]
        
        NewNumber = patch + Error * ones * kernel
        NewNumber[NewNumber>1] = 1
        NewNumber[NewNumber<0] = 0
        
        result4[y:y+3, x-2:x+3] = NewNumber
        result4[y, x] = new_value
            
result4 = result4[2:258, 2:258]

# p2 (a)
ratio = 0.5
result5 = cv2.resize(sample2, dsize=[0,0], fx=ratio, fy=ratio)

ratio = 1/ratio
result5 = cv2.resize(result5, dsize=[0,0], fx=ratio, fy=ratio)

# p2 (b)
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    return 1-gaussianLP(D0,imgShape)

d0 = 5
original = np.fft.fft2(sample3) #傅立葉轉換
center = np.fft.fftshift(original) #將座標(0,0)轉到中心
HighPassCenter = center * gaussianHP(d0,sample3.shape) #乘以 high pass mask
HighPass = np.fft.ifftshift(HighPassCenter) #將座標轉回去
inverse_HighPass = np.fft.ifft2(HighPass) #做逆傅立葉轉換

result6 = np.abs(inverse_HighPass)
result6 = (result6-result6.min())/(result6.max()-result6.min())*255

cv2.imwrite("result1.png",result1*255)
cv2.imwrite("result2.png",result2*255)
cv2.imwrite("result3.png",result3*255)
cv2.imwrite("result4.png",result4*255)
cv2.imwrite("result5.png",result5)
cv2.imwrite("result6.png",result6)
