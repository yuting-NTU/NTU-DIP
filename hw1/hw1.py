import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import time

def polt2img(img1, img2, name1, name2, gray=True):
    plt.figure(figsize=(12,4))

    plt.subplot(121)
    if gray:
        plt.imshow(img1, cmap='gray',vmin=0, vmax=255)
    else:
        plt.imshow(img1)
    plt.title(name1)

    plt.subplot(122)
    if gray:
        plt.imshow(img2, cmap='gray',vmin=0, vmax=255)
    else:
        plt.imshow(img2)
    plt.title(name2)

    plt.show()

def plot_img_histogram(img, title):
    hist,bins = np.histogram(img,bins=256,range=(0,256))
    bins = bins[:-1]
    plt.bar(bins,hist)
    plt.title(title)

def plot_img_cdf_pdf(img, name):
    hist,bins = np.histogram(img,bins=256,range=(0,256))
    pdf = hist/img.size
    cdf = pdf.cumsum() 

    plt.figure(figsize=(12,4)) #圖要設大一點柱狀圖才不會被壓縮變形

    plt.subplot(131)
    plt.bar(bins[:-1],hist)
    plt.title('histogram of {}'.format(name),fontsize=24)

    plt.subplot(132)
    plt.bar(bins[:-1],pdf)
    plt.title('pdf of {}'.format(name),fontsize=24)

    plt.subplot(133)
    plt.bar(bins[:-1],cdf)
    plt.title('cdf of {}'.format(name),fontsize=24)

    plt.show()

def globalHE(img):
    hist,bins = np.histogram(img.ravel(),256,[0,255])
    pdf = hist/img.size # 出現次數/總像素點 = 機率 = pdf
    cdf = pdf.cumsum() # 將機率利用cumsum()累加 = cdf
    equ_value = np.around(cdf * 255).astype('uint8') #將cdf乘以255(max value) ，再四捨五入取整數
    result = equ_value[img] #將原本的value數值transfer到對應的數值
    return result

def localHE(img, kernel_size):
    print("\nstart AHE, kernel_size=",kernel_size)
    kernel_size_squared = int(kernel_size * kernel_size)
    # Padding
    img = np.lib.pad(img,(kernel_size,kernel_size),'reflect')
    # 設定灰階的最大值
    max_value = 255
    # 開一個新的陣列存 AHE img
    ime_ahe = np.zeros_like(img)
    # start
    t_start = time.time()
    # 遍歷圖片的每個pixels
    for i in range(0,img.shape[0]-kernel_size):
        for j in range(0,img.shape[1]-kernel_size):
            #提取圖片的區塊(kernel_size x kernel_size)
            kernel = img[i:i+kernel_size,j:j+kernel_size]
            #由小到大排序
            kernel_flat = np.sort(kernel.flatten())
            #找到目前的pixel在這個區塊排第幾名
            rank = np.where(kernel_flat == img[i,j])[0][0] 
            #排第幾名就獲取相對應排名的亮度
            ime_ahe[i,j] = int( max_value * ( rank / kernel_size_squared ) )
            
    # end
    t_end = time.time()
    print ('Total time taken in seconds: ',(t_end-t_start))
    # 將之前的padding切掉
    img_sliced = ime_ahe[(0+kernel_size):(img.shape[0]-kernel_size),(0+kernel_size):(img.shape[1]-kernel_size)]
    image_output = np.array(img_sliced, dtype = np.uint8)
    
    return image_output

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2) 
    return 10 * math.log(math.pow(255, 2) / mse, 10)

sample1 = cv2.imread('hw1_sample_images/sample1.png')
sample1 = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)
sample2 = cv2.imread('hw1_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
sample3 = cv2.imread('hw1_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
sample4 = cv2.imread('hw1_sample_images/sample4.png', cv2.IMREAD_GRAYSCALE)
sample5 = cv2.imread('hw1_sample_images/sample5.png', cv2.IMREAD_GRAYSCALE)

# P0 a 
height, width, channel = sample1.shape
result1 = sample1.copy()
# swap by column
for j in range(width//2):
    result1[:, [j, width-j-1]] = result1[:, [width-j-1, j]]
# polt2img(sample1, result1, 'sample1', 'result1', gray=False)

# P0 b
result2 = result1.copy()
result2 = 0.299*result2[:,:,0]+0.587*result2[:,:,1]+0.114*result2[:,:,2]
# polt2img(result1, result2, 'result1', 'result2')

# P1 a
result3 = sample2.copy()
result3 = result3/2 
result3 = np.around(result3)
# polt2img(sample2, result3, 'sample2', 'result3')

# P1 b
result4 = result3.copy()+1 #避免乘到0還是0
result4 = result4*3.0
result4 = np.around(result4)
result4 = np.clip(result4,0,255)

# P1 c
plt.figure(figsize=(12,4))

plt.subplot(131)
plot_img_histogram(sample2, 'histograms of sample2')

plt.subplot(132)
plot_img_histogram(result3, 'histograms of result3')

plt.subplot(133)
plot_img_histogram(result4, 'histograms of result4')

plt.tight_layout()
plt.show()

# P1 d
result5 = globalHE(result3.astype('uint8')) 
result6 = globalHE(result4.astype('uint8')) 
plot_img_cdf_pdf(result5, 'result5')
plot_img_cdf_pdf(result6, 'result6')

# P1 e
result7 = localHE(result3, 30)
result8 = localHE(result4, 30)
plot_img_cdf_pdf(result7, 'result7')
plot_img_cdf_pdf(result8, 'result8')

# P1 f
result9 = sample2.copy()
result9 = np.power((result9 / 255), 0.5) * 255
plot_img_cdf_pdf(result9, 'result9')

# P2 a

# Gaussian noise
# b = 30
# kernel = (np.array([[1, b, 1] ,[b, math.pow(b, 2), b] ,[1, b, 1]])) / math.pow(b + 2, 2)
# kernel_size = kernel.shape[0]
kernel_size = 5
kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)

# Padding
result10 = np.lib.pad(sample4, kernel.shape, 'reflect')
height, width = result10.shape
# print(height, width)
gap = kernel_size // 2
for n in range(1):
    for i in range(gap, height - gap):
        for j in range(gap, width - gap):
            #將位置對應的pixel與kernel相乘後取sum
            p = (result10[i-gap:i+gap+1,j-gap:j+gap+1] * kernel).sum()
            result10[i,j] = p

# 將padding切掉
result10 = result10[kernel_size:height-kernel_size,kernel_size:width-kernel_size]


# Salt-and-pepper noise
kernel_size = 3
# Padding
result11 = np.lib.pad(sample5, (kernel_size,kernel_size), 'reflect')
height, width = result11.shape
gap = kernel_size // 2
for i in range(gap, height - gap):
    for j in range(gap, width - gap):
        #取kernel的中位數
        result11[i,j] = np.median(result11[i-gap:i+gap+1,j-gap:j+gap+1])
        
# 將padding切掉
result11 = result11[kernel_size:height-kernel_size,kernel_size:width-kernel_size]


# P2 b
print('\npsnr:')
print('result10:', psnr(sample3,result10))
print('result11:', psnr(sample3,result11))

# save img
result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
cv2.imwrite("result1.png",result1)
cv2.imwrite("result2.png",result2)
cv2.imwrite("result3.png",result3)
cv2.imwrite("result4.png",result4)
cv2.imwrite("result5.png",result5)
cv2.imwrite("result6.png",result6)
cv2.imwrite("result7.png",result7)
cv2.imwrite("result8.png",result8)
cv2.imwrite("result9.png",result9)
cv2.imwrite("result10.png",result10)
cv2.imwrite("result11.png",result11)