import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import os


data0 = 'D:/8P361 Project BIA/train+val/train/0'
data1 = 'D:/8P361 Project BIA/train+val/train/1'

fig, (ax0,ax1) = plt.subplots(1, 2) 
ax0.set_title("No metastase")
ax1.set_title("With metastase")

img0 = Image.open(data0)
img1 = Image.open(data1)

ax0.imshow(img0)
ax1.imshow(img1)
display.clear_output(wait=True)
display.display(plt.gcf())

