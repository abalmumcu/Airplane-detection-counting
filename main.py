
import cv2
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology, io
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_closing, disk
import scipy.ndimage as nd
import math

# Görüntünün tanımlanması
img = cv2.imread('/content/drive/MyDrive/planes/test/gray_30.tif',0) 

# Görüntünün yeniden boyutlandırılması

src = img
scale_percent = 20

# Orjinal görüntünün %20'si alınmış. 
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# yeniden oluşturulan görüntü
img = cv2.resize(src, dsize)

# Median Filtresi
img_blur = cv2.medianBlur(img,3)
cv2.imwrite("blured.pgm",img_blur)


def auto_canny(image, sigma=0.1):
	# Görüntünün medyanı alındı
	v = np.median(image)
	# medyan değeri kullanılarak lower ve upper threshold değeri saptandı
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# oluşturulan görüntü döndürülüyor
	return edged

# Fonksiyonu oluşturulan Canny'ye medyan filtresi sonucu veriliyor
canny_img = auto_canny(img_blur)
canny_img.dtype
cv2.imwrite("canny.pgm",canny_img)


# dilation işlemi (3,3) elips yapısal elemanıyla bulunuyor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilated = cv2.dilate(canny_img, kernel,iterations=1) 
cv2.imwrite("dilated.pgm",dilated)

# Görüntünün etiketleri çıkartıldı
retval, labels = cv2.connectedComponents(dilated)
# Eğer toplam sayılan piksel sayısı N değerinden küçükse, o label'a 0 ata.
num = labels.max()
N = 500
for i in range(1, num+1):
    pts =  np.where(labels == i)
    if len(pts[0]) < N:
        labels[pts] = 0



# Çıkarılan görüntünün siyah arka plana atanması
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# Görüntülemek için HSV formatından BGR'a çevrilmiştir.   
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# Arka plan siyah yapıldı
labeled_img[label_hue==0] = 0

cv2.imwrite("labeled_img_rgb.png",labeled_img)

# Closing algoritmasına sokmak için tekrar Grayscale formata çevrildi
labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_BGR2GRAY)

# Closing işlemi mask ile yapıldı
strel = disk(4)
I_closed = binary_closing(labeled_img, strel)
# İçerisinin tamemen dolması için morfology uygulandı
I_closed_filled = nd.morphology.binary_fill_holes(I_closed)

I_closed_filled.dtype='uint8'
I_closed_filled.dtype


# görüntünün etiketleri çıkartılarak öznitelik çıkarımı için regionprops fonksiyonuna verildi
label_img = measure.label(I_closed_filled)
regions = measure.regionprops(label_img)
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)
k = 1
# Eğer bulunan alan 6500'den büyükse centroidini al ve orayı dikdörtgen içine al
# Orta noktasına obje numarasını ata
# Area ve eccentricity değerini yazdır.
count = []

for props in regions:
    if (props.area > 6500):
      y0, x0 = props.centroid
      orientation = props.orientation
      x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
      y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
      x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
      y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

      plt.text(x0, y0, "{}".format(k), bbox=dict(facecolor='red', alpha=0.5))

      k = k+1
      minr, minc, maxr, maxc = props.bbox
      bx = (minc, maxc, maxc, minc, minc)
      by = (minr, minr, maxr, maxr, minr)
      ax.plot(bx, by, '-b', linewidth=2.5)
      print('Label: {} >> Object area: {}, Eccentricity: {} '.format(props.label, props.area, props.eccentricity))
      count.append(orientation)
plt.text(40, 40, "{} Plane Detected".format(len(count)), bbox=dict(facecolor='green', alpha=0.5),fontsize=10)
# plt.title('Label: {} >> Object area: {}, Eccentricity: {} '.format(props.label, props.area, props.eccentricity)
ax.axis('off')
plt.savefig("output.png",)
plt.show()