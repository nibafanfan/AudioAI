# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:16:12 2022

@author: nibaf
"""

#!/usr/bin/env python
# coding: utf-8


# In[1]:

# Julian Loiacono wuz here

from scipy import signal
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
import sys
rng = np.random.default_rng()


here = os.path.dirname(os.path.abspath(__file__))

# In[2]:


def spectrogram_for_random_snippet(y,fileid):
    dur=5*44100
    I=np.random.choice(range(len(y)-dur))
    s=y[I:I+dur]
    nperseg=1024
    f, t, Sxx = signal.spectrogram(s, fs=44100, nperseg=nperseg)
    return(f,t,Sxx)


# In[32]:


# just tensors stacked& y is numbered
import re

nepoch= int(input("Enter nepoch: "))
nbatch= int(input("Enter nbatch: "))
ncount= input("Enter ncount(no more than # of samples in each genre): ")
ncount=int(ncount)
print(ncount, type(ncount),'print')

g=-1
X=[]
Y=[]
genrelst=['Bachata','Tango']
# genrelst=['Afro Hits','Bachata', 'Disney Hits', 'Electronic_ Definitive', 'Hiphop', 
            # 'Saxophone Hits Cover', '1990s R&B', '2010s R&B', 'lofi + chill','Peaceful Piano','Blues Classics', 'Jazz Classics']
nclass=len(genrelst)
for genre in genrelst:
#     print(genre)
    g+=1
    count=0
    L=os.listdir(os.path.join(here, genre))
    L=[x for x in L if re.search("\.wav$",x)]
    for filename in L:
        fileid=genre+"\{0}".format(filename)
        print(fileid)
        try:
            
            x,samplerate=sf.read(genre+'/'+filename)
#             print(samplerate)
            
            #
            # make this mono by adding the two channels
            #
            y=x[:,0]+x[:,1]
            y.shape
            f,t,Sxx=spectrogram_for_random_snippet(y,fileid)

            Tri_Sxx=np.log(Sxx)
            X.append(Tri_Sxx)
            Y.append(g)
            count+=1
            
            if count>ncount:
                break
        except:
            print('exception occurred', fileid)
            pass
            
# In[33]:


lenX=len(X)
print('lenX', lenX)
print(X[0].shape, 'X[0] shape')
print(len(Y), 'len(Y)')
print(Y[1], 'Y[1]')
print(Y, "Y")
print(type(X))
print(type(Y))

np.save('X_'+str(nclass)+'limit'+str(ncount),X)
np.save('Y_'+str(nclass)+'limit'+str(ncount),Y)

#%%


X2=[]
X=np.load('X_'+str(nclass)+'limit'+str(ncount)+'.npy')
Y=np.load('Y_'+str(nclass)+'limit'+str(ncount)+'.npy')

for i in X:
    print('i in X', i)
    temp=[]
    temp2=[]
    for j in i:
        
        temp.append(np.max(i))
        temp2.append(np.min(i))
    maxt=max(temp)
    mint=min(temp2)
    if maxt==mint:
        print('max and min equal')
        X2.append(i-mint)
    else:
        X2.append((i-mint)/(maxt-mint))

# for i in X:
#     X2.append((i-mint)/(maxt-mint))    




#%%
print(X2, "X2")




# In[39]:

    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.1, random_state=42)
    

print(len(X_train),len(X_test),len(y_train),len(y_test))




# In[40]:


np.save('MusicTrainingImages'+str(nclass)+'limit'+str(ncount),X_train)
np.save('MusicTrainingLabels'+str(nclass)+'limit'+str(ncount),y_train)


# In[41]:


np.save('MusicTestingImages'+str(nclass)+'limit'+str(ncount),X_test)
np.save('MusicTestingLabels'+str(nclass)+'limit'+str(ncount),y_test)


# In[42]:


X=np.load('MusicTrainingImages'+str(nclass)+'limit'+str(ncount)+'.npy')
lenX=len(X)
Xnew=np.zeros((lenX,1,513,245))
for i in range(lenX):
    Xnew[i,0,:,:]=X[i,:,:]
np.save('MusicTrainingImagesReshaped'+str(nclass)+'limit'+str(ncount),Xnew)

X=np.load('MusicTestingImages'+str(nclass)+'limit'+str(ncount)+'.npy')
lenX=len(X)
Xnew=np.zeros((lenX,1,513,245))
for i in range(lenX):
    Xnew[i,0,:,:]=X[i,:,:]
np.save("MusicTestingImagesReshaped"+str(nclass)+'limit'+str(ncount),Xnew)

#%%
#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Similar to Resnet for Music, report target out of bound at loss=criterion(outputs,y)
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device="cpu"
torch.cuda.empty_cache() 

class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has '                                                      f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


class MyDataSet(Dataset):
        def __init__(self,X,Y):
            self.X=X
            self.Y=Y
            self.N=self.X.shape[0]
            self.K=self.X.shape[1]
        def __len__(self):
            return(self.N)
        def __getitem__(self, idx):
            return self.X[idx],self.Y[idx]

Xtrain=np.load("MusicTrainingImagesReshaped"+str(nclass)+'limit'+str(ncount)+".npy")
Ytrain=np.load("MusicTrainingLabels"+str(nclass)+'limit'+str(ncount)+".npy")
# print(Ytrain)


# # In[7]:


Xtest=np.load("MusicTestingImagesReshaped"+str(nclass)+'limit'+str(ncount)+".npy")
Ytest=np.load("MusicTestingLabels"+str(nclass)+'limit'+str(ncount)+".npy")

ntest=Xtest.shape[0]
print("data read from disk")

TrainingData=MyDataSet(Xtrain,Ytrain)
TestingData=MyDataSet(Xtest,Ytest)

print("created training & testing data")


DataLoader_train=torch.utils.data.DataLoader(
    dataset=TrainingData,
    batch_size=nbatch,
    shuffle=True)

print("created data loader")

learning_rate = .01
num_epochs=nepoch

model = ResNet50(img_channels=1, num_classes=nclass)

print("created model")


#
# move the model to the device
#
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
#
# Adam is an optimizer that maintains a different learning 
# rate for every weight/parameter in the network - so the learning rate 
# is a relative one
#
# See the gentle introduction:
#        e.g. https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    print("epoch = "+str(epoch))
    #
    # for each epoch the data loader loads the data one batch at a time
    #
    for i,(x,y) in enumerate(DataLoader_train):
        #if i>1:
        #    break
        #print(x.shape)
        #print("i = "+str(i))
        x=x.float()
        x=x.to(device)
        y=y.to(device,dtype=torch.long)

        optimizer.zero_grad()

        #print(x.size())
        outputs=model(x)

        loss=criterion(outputs,y)
        loss_value = loss.item()

        loss.backward() # compute gradients

        # move in direction of minus the gradient by a learning_rate amount 
        # here because we are using Adam, step is more complicated than -epsilon*Gradient
        optimizer.step() 
        #print("epoch = {0:5d} i = {1:5d} loss = {2:8.5f}".format(epoch,i,loss_value))
    print("epoch = {0:5d} loss = {1:8.5f}".format(epoch,loss_value))
#
nclasses=nclass
Confusion=np.zeros(shape=(nclasses,nclasses),dtype=int)
xtest=torch.tensor(Xtest).float().to(device)
ypred=model(xtest)
ypred=ypred.cpu().detach().numpy()
ypred=np.apply_along_axis(np.argmax,1,ypred)

# In[ ]:

# from sklearn.metrics import confusion_matrix as confusion_matrix
# Confusion = confusion_matrix(Ytest, ypred)
# s=0
# for i in range(len(xtest)):
#     Confusion[Ytest[i],ypred[i]]+=1
#     s+=(Ytest[i]==ypred[i])
# accuracy=s/len(xtest)


# In[ ]:





# In[4]:


# s


# In[5]:


# s/10000
# #%%
# print(Confusion,'Confusion')
# print(accuracy,'accuracy')
# print(s)


# # In[ ]:
# np.save('Confusion_'+str(nclass)+'limit'+str(ncount)+'b'+str(nbatch)+'ep'+str(nepoch), Confusion)
# np.save('s_'+str(nclass)+'limit'+str(ncount)+'b'+str(nbatch)+'ep'+str(nepoch), s)




