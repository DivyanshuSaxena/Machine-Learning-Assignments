#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import math
import random
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# In[4]:


def load_image(infilename):
    img = Image.open(infilename)
    data = np.array(img)
    return data

def view_image(oned_arr):
    image = np.reshape(oned_arr, (3, 160, 210))
    print (image.shape)
    image = np.swapaxes(image, 0, 2)
    img = Image.fromarray(image)
    plt.imshow(img)
    plt.show()

gen_data = False

if gen_data:
    rootdir = "./train_dataset/"
    tree = sorted(list(os.walk(rootdir)))


# In[8]:


if gen_data:
    for root, sub_folders, files in tree:
        print (root)
        if len(files) != 0:
            episode_images = []
            reward_file = os.path.join(root, 'rew.csv')
            episode_reward = np.genfromtxt(reward_file)
            index = 0
            file_index = 0
            for file in sorted(files)[0:-1]:
                image_file = os.path.join(root, file)
                image_arr = load_image(image_file)
                image_arr = np.swapaxes(image_arr, 0, 2)
                episode_images.append(image_arr)
                index += 1
                if index == 500:
                    index = 0
                    episode_images = np.array(episode_images)
                    print ("{0} {1}".format(root[16:], file_index))
                    np.save('./episodes/{0}_{1}'.format(root[16:], file_index), episode_images)
                    episode_images = []
                    file_index += 1
            if not len(episode_images) == 0:
                episode_images = np.array(episode_images)
                print ("{0} {1}".format(root[16:], file_index))
                np.save('./episodes/{0}_{1}.npy'.format(root[16:], file_index), episode_images)


# In[4]:


episode = np.load('./episodes/00000500_2.npy')
print (episode.shape)
# view_image(episode[25])


# In[6]:


if gen_data:
    rewards = []
    for root, sub_folders, files in tree:
            print (root)
            if len(files) != 0:
                reward_file = os.path.join(root, 'rew.csv')
                episode_reward = np.genfromtxt(reward_file)
                rewards.append(episode_reward)
    rewards = np.array(rewards)
    print (rewards.shape)
    print (rewards[0].shape)
    
    segment_rewards = []
    index = 0
    for i_reward in rewards:
        print (index)
        episode_rewards = []
        num_segments = len(i_reward)//500 + 1
        for segment in range(num_segments):
            if (segment+1)*500 < len(i_reward):
                seg_reward = i_reward[segment*500+6:(segment+1)*500]
            else:
                seg_reward = i_reward[segment*500+6:]
            episode_rewards.append(seg_reward)
        episode_rewards = np.array(episode_rewards)
        segment_rewards.append(episode_rewards)
        index += 1
    segment_rewards = np.array(segment_rewards)
    print (segment_rewards.shape)
    np.save('./episodes/seg_rewards.npy', segment_rewards)


# In[4]:


if not gen_data:
    segment_rewards = np.load('./episodes/seg_rewards.npy')
    print (segment_rewards.shape)
    print (segment_rewards[0].shape)


# In[30]:


test = True
if test:
    # Load Test Data
    rootdir = "./test_dataset/"
    tree = sorted(list(os.walk(rootdir)))
else:
    # Load Validation Data
    rootdir = "./validation_dataset/"
    tree = sorted(list(os.walk(rootdir)))


# In[31]:


index = 0
file_index = 0
test_data = []
if test:
    save_name = "test"
else:
    save_name = "val"
for root, sub_folders, files in tree[1:]:
    print (root)
    sample = []
    if len(files) != 0:
        for file in sorted(files):
            image_file = os.path.join(root, file)
            image_arr = load_image(image_file)
            image_arr = np.swapaxes(image_arr, 0, 2)
            sample.append(image_arr)        
        test_data.append(sample)
        index += 1
        if index == 1000:
            index = 0
            test_data = np.array(test_data)
            np.save('./neuralset/{0}_{1}'.format(save_name, file_index), test_data)
            test_data = []
            file_index += 1
if not len(test_data) == 0:
    test_data = np.array(test_data)
    np.save('./neuralset/{0}_{1}.npy'.format(save_name, file_index), test_data)


# In[28]:


if not test:
    reward_file = './validation_dataset/rewards.csv'
    val_rewards = np.genfromtxt(reward_file, delimiter=',')
    val_rewards = val_rewards[:, -1]
    print (val_rewards.shape)


# In[6]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BreakoutDataset(Dataset):
    """Dataloader class for Breakout"""

    def __init__(self, num_samples, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.num_samples = num_samples
        dataset = []
        per_iter = 50
        num_iters = num_samples//per_iter
        for i in range(num_iters):
            episode_num = random.randint(1, 500)
            num_segments = len(segment_rewards[episode_num-1])
            segment_num = np.random.randint(0, num_segments)
            segment_reward = segment_rewards[episode_num-1][segment_num]
            if len(segment_reward) < 50:
                segment_num -= 1
                segment_reward = segment_rewards[episode_num-1][segment_num-1]
            episode_file = './episodes/{:08d}_{:d}.npy'.format(episode_num, segment_num)
            # print ('Reading ep: {0}, segment: {1}'.format(episode_num, segment_num))
            segment = np.load(episode_file)

            # Take reward 1 sample with prob sample_one and 0 o.w.
            sample_one = 0.3
            prob = np.random.uniform()
            if prob < sample_one:
                # print ('Choosing one')
                sample_reward = 1
                indices = np.where(segment_reward==1)[0]
            else:
                # print ('Choosing zero')
                sample_reward = 0
                indices = np.where(segment_reward==0)[0]
            
            # Given the segment, get per_iter number of frames from it
            for j in range(per_iter):
                f_index = np.random.choice(indices)
                reward = segment_reward[f_index]
                frame_indexes = np.sort(np.random.choice(6, 4, replace=False))
                frame = []
                for index in frame_indexes:
                    frame.append(segment[f_index-6+index])
                frame.append(segment[f_index])
                frame = np.array(frame)
                frame = frame.reshape((-1, ) + frame.shape[2:])
                input_tensor = torch.from_numpy(frame)
                label_tensor = torch.from_numpy(np.array([sample_reward]))
                sample = {'sample': input_tensor, 'reward': label_tensor}
                dataset.append(sample)
            print (len(dataset))
        self.dataset = dataset

        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample


# In[7]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(15, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64*9*12, 2048)
        self.fc2 = nn.Linear(2048, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net().to(device)
print(net)


# In[8]:


input = torch.randn(4, 15, 160, 210)
out = net(input.to(device))
print(out)


# In[35]:


criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[8]:


test_dataset = BreakoutDataset(20000)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=4)


# In[36]:


for i in range(2):
    train_dataset = BreakoutDataset(20000)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
    print ("Dataset generated")
    classes = [0, 1]
    
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i_batch, sample_batched in enumerate(train_loader):
            print(i_batch, sample_batched['sample'].size(), sample_batched['reward'].size())

            inputs = sample_batched['sample']
            labels = sample_batched['reward']

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            labels = labels.squeeze_()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i_batch % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, running_loss / 20))
                running_loss = 0.0
                
        torch.save(net.state_dict(), "./models/nn_model" + ".pt")

print('Finished Training')


# In[37]:


correct = 0
total = 0
pred = []
actual = []
with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_loader):
        inputs = sample_batched['sample']
        labels = sample_batched['reward']
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.squeeze_()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        preds_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()
        actual.append(labels_np)
        pred.append(preds_np)

actual = np.array(actual)
pred = np.array(pred)
print (pred.shape)
print (actual.shape)
pred = pred.flatten()
actual = actual.flatten()
print ('Accuracy of the network on the test set: {0}%'.format(100 * correct/total))
f1 = f1_score(actual, pred)
print ("F1 Score: {0}".format(f1))


# In[15]:


net.load_state_dict(torch.load('./models/nn_model.pt'))
net.eval()


# In[16]:


scratch = "/scratch/cse/btech/cs1160310/np_testset_CNN/"
l1 = os.listdir(scratch)
l1.sort()
out = []
with torch.no_grad():
    for i in range(0,10):
        # print i
        images = np.load("/scratch/cse/btech/cs1160310/np_testset_CNN/" + l1[i])
        print ("image shape: {0}".format(images.shape))
        for j in range(0,len(images)):
            print (i,j)
            l = []
            l.append(images[j])
            arr = np.array(l)
            inputs = torch.from_numpy(arr)
            inputs = inputs.type(torch.FloatTensor)
            outputs = net(inputs.to(device))
            if (outputs[0][0] > outputs[0][1]):
                out.append(0.0)
            else:
                out.append(1.0)
            # print outputs
            # print labels[i]
ly=len(out)
with open("kaggleresult.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(["id","Prediction"]) 
    for i in range(ly):
        csvwriter.writerow([i,int(out[i])])

