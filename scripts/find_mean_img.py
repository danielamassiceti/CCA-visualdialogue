import cv2, torch, os
from PIL import Image
from torchvision import transforms

directory = '' # set image directory here
if directory == '':
    print ('set directory variable to location of images!')
    exit()
totensor = transforms.ToTensor()

c = 0
running_mean = torch.zeros(3)
running_std = torch.zeros(3)
all_means = []
for imgname in os.listdir(directory):
    img = totensor(Image.open(os.path.join(directory, imgname)).convert('RGB'))
    all_means.append(torch.mean(img.view(3,-1), 1))
    running_mean += all_means[-1]
    tmp = torch.mean(img.view(3,-1), 1) - means
    stds += tmp.pow(2)
    c += 1

running_mean /= c
for cc in xrange(c):
   tmp = all_means[cc] - running_mean
   running_std += tmp.pow(2)

stds1 = torch.sqrt(running_std/c)
stds2 = torch.sqrt(running_std/(c-1))

print ('Mean image: ')
print(running_mean)
print ('Standard deviation (N): ')
print (stds1)
print ('Standard deviation (N-1): ')
print (stds2)
