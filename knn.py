from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms, datasets
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#data processing
class GenderDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, image_dir, transformss,flag):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transformss = transformss
        self.flag=flag

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if self.flag=='train':
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.image_dir,
                                str(self.csv_file.iloc[idx, 0])+'.jpg')
            image = Image.open(img_name)
#             image = io.imread(img_name)
            label = self.csv_file.iloc[idx, 1]
            label = np.array([label])
            label = label.astype('float')
#             label = label.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': label}

            image=sample['image']
            label=sample['landmarks']
        
#             print(type(image))
            image =self.transformss(image)

            return {'image': image, 'landmarks': label}
       
        if self.flag=='test':
            if torch.is_tensor(idx):
                idx = idx.tolist()
  
            img_name = os.path.join(self.image_dir,
                                str(self.csv_file.iloc[idx, 0])+'.jpg')
            image = Image.open(img_name) #提取图片
            imageid=self.csv_file.iloc[idx, 0]
#           image = io.imread(img_name)
#           label = self.csv_file.iloc[idx, 1]
#           label = np.array([label])
#           label = label.astype('float')
#           label = label.astype('float').reshape(-1, 2)
            sample = {'imageid': imageid, 'image': image}
            image=sample['image']  
            image =self.transformss(image)

            return {'imageid': imageid, 'image': image}

train_img_dir='/kaggle/input/gender/train/train'
train_csv_file='/kaggle/input/gender/train.csv'

test_img_dir='/kaggle/input/gender/test/test'
test_csv_file='/kaggle/input/gender/test.csv'

train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
test_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
train_dataset=GenderDataset(train_csv_file,train_img_dir,train_data_transform,'train')
test_dataset=GenderDataset(test_csv_file,test_img_dir,test_data_transform,'test')

if torch.cuda.is_available():
    device = torch.device('cuda:0')#选择GPU
    print("GPU is open and connection") 
    
else:
    print('no GPU')
    
# 建立训练集矩阵
temp=train_dataset.__getitem__(0)
temp=temp['image'].reshape(-1, 200 * 200 * 3) #二维矩阵
temp=temp[0]  #一维矩阵
for i in range(1,5000):
    temp_temp=train_dataset.__getitem__(i)
    temp_temp=temp_temp['image'].reshape(-1, 200 * 200 * 3)
    temp_temp=temp_temp[0]
    
    temp=np.vstack((temp,temp_temp))
    
#     temp=temp.to(device)
    
    if i%1500==0:
        print("1500次")
        
# print(temp.shape)
# print(type(temp)) #此时的temp是numpy类型
# temp=torch.from_numpy(temp)
# temp=temp.to(device)
# print(type(temp))
#降维
from sklearn.decomposition import PCA #PCA要求降维后的特征数必须小于样本数
pca=PCA(200)

# temp=train_dataset.__getitem__(0)
# temp=temp['image'].reshape(3, 200*200)
# temp=temp[0]

# print(temp.shape)
pca.fit(temp)
Xtrain=pca.transform(temp)
print(Xtrain.shape) 
print(type(Xtrain))
Xtrain=torch.from_numpy(Xtrain)
Xtrain.to(device)
# 训练集维度（3000*200） 
# 此时已将训练集放入到GPU上

# 在测试集上降维
temp_test=test_dataset.__getitem__(0)
temp_test=temp_test['image'].reshape(-1, 200 * 200 * 3) #二维矩阵
temp_test=temp_test[0]  #一维矩阵

for i in range(1,300):
    temp_temp=train_dataset.__getitem__(i)
    temp_temp=temp_temp['image'].reshape(-1, 200 * 200 * 3)
    temp_temp=temp_temp[0]
    
    temp_test=np.vstack((temp_test,temp_temp))
    
    if i%1500==0:
        print("1500次")
    
#     temp_test.to(device)
    
print(temp_test.shape)

#降维
from sklearn.decomposition import PCA #PCA要求降维后的特征数必须小于样本数
pca=PCA(200)

# temp=train_dataset.__getitem__(0)
# temp=temp['image'].reshape(3, 200*200)
# temp=temp[0]

# print(temp.shape)
pca.fit(temp_test)
temp_test=pca.transform(temp_test)
print(temp_test.shape)

temp_test=torch.from_numpy(temp_test)
temp_test.to(device)

# 此时已将测试集放置到GPU上

# 获得训练集的标签
train_label=[]
for i in range(5000):
    label=train_dataset.__getitem__(i)['landmarks']
    train_label.append(label)
print(len(train_label))

# train_label.to(device) 

# 获得测试集的标签
test_label=[]
for i in range(300):
    label=train_dataset.__getitem__(i)['landmarks']
    test_label.append(label)
print(len(test_label))

# test_label.to(device)
train_label=torch.tensor(train_label)
test_label=torch.tensor(test_label)

print(type(train_label))

from sklearn.neighbors import KNeighborsClassifier
kNN_classifier = KNeighborsClassifier(n_neighbors=5)

# kNN_classifier.to(device)

kNN_classifier.fit(Xtrain, train_label)
print(type(kNN_classifier))

pred=kNN_classifier.predict(temp_test)
print(len(pred))
print(pred)

# 提交结果
test_csv = pd.read_csv('/kaggle/input/gender/test.csv')
submit = pd.DataFrame({'Id':test_csv.id[:300],'label':pred})
submit.to_csv("submission.csv",index=False)
print(submit)
