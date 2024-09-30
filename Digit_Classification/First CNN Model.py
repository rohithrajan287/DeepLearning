'''
Steps to Build a CNN for Digit Classification (0-9):
----------------------------------------------------
1.Load the MNIST Dataset.
2.Preprocess the Data (normalize and reshape).
3.Define the CNN Architecture.
4.Train the Model on the training data.
5.Evaluate the Model on the test data.

-----------------------------------------------------------------------------------------------------------------------------------
1.Import the Library:
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

'''
2.Load the MNIST Dataset.
'''
#Load The DataSet
#Transform used in Loading data set Function
#First we are converting images to tensors(Multi_dimensional Array) and Then normalizing it (-1,1)
#It might have 0-255 value, mormalizing it helps in accuracy
transform = transforms.Compose
([ transforms.ToTensor(),
  transforms.Normalize((0.5),(0.5))
 ])
#[ToTensor(), Normalize(mean=0.5, std=0.5)]

# Will use pre-loaded data set (MINST). root -->directory train(will define train data or test data)
# transform function will be called to process the picked images
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#To Load the images, here we define the no of images in each batch
#shuffling helpes to improve accuracy
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

print(f'Training dataset size: {len(trainLoadSet)}')
print(f'Testing dataset size: {len(testLoadSet)}')

####OUTPUT###
#  raining dataset size: 1875
#  Testing dataset size: 313

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Added 3 convolution layers to improve accuracy)
        #greyscale image has 1 input and out channels depend on the filter used
        #padding - extra pixel added to image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  

        #output matrix divided into 2*2 for tiled-matrix operations
        self.pool = nn.MaxPool2d(2, 2) 

        #Now the tiled matrix is passed to fully_connected layers to get output
        #here we pass all the cnn layer generated matrix and define the out feautures(128 feautre maps with 3*3 size)
        #the next layer defines that 10 diff class are in the 128 out features of previous layer
        self.fc1 = nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
       #pooling reduces the dimension of formed matrix
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x))) 

        # multi - dimensional array to one dimensional array
        x = x.view(-1, 128 * 3 * 3)

        # call fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

# Initiate the model(defined earlier) and defining of loss function
model = CNN()
criterion = nn.CrossEntropyLoss()
#learning rate will be increased to 0.01 to increase speed
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function(this calls all the initialized values)
def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad() 
            #now we pass our input to the model
            outputs = model(inputs)  
            #based on the output loss is computed and bias is altered
            loss = criterion(outputs, labels)  
            #now passed back again for best accuracy
            loss.backward()  
            #weights are updated here
            optimizer.step()  

            running_loss += loss.item()
            #to print statistics every 100 batches
            #total 5 epochs---36 image in a batch
            if i % 100 == 99:  
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
               
train_model(model, trainloader, criterion, optimizer, num_epochs=5)

####OUTPUT####
'''
Epoch [1/5], Step [100/1875], Loss: 1.6289
Epoch [1/5], Step [200/1875], Loss: 0.4725
Epoch [1/5], Step [300/1875], Loss: 0.3510
Epoch [1/5], Step [400/1875], Loss: 0.3183
Epoch [1/5], Step [500/1875], Loss: 0.2609
Epoch [1/5], Step [600/1875], Loss: 0.2363
Epoch [1/5], Step [700/1875], Loss: 0.2164
Epoch [1/5], Step [800/1875], Loss: 0.2264
Epoch [1/5], Step [900/1875], Loss: 0.2022
Epoch [1/5], Step [1000/1875], Loss: 0.1917
Epoch [1/5], Step [1100/1875], Loss: 0.2151
Epoch [1/5], Step [1200/1875], Loss: 0.2213
Epoch [1/5], Step [1300/1875], Loss: 0.2097
Epoch [1/5], Step [1400/1875], Loss: 0.2117
Epoch [1/5], Step [1500/1875], Loss: 0.1785
Epoch [1/5], Step [1600/1875], Loss: 0.1877
Epoch [1/5], Step [1700/1875], Loss: 0.1841
Epoch [1/5], Step [1800/1875], Loss: 0.1992
Epoch [2/5], Step [100/1875], Loss: 0.1461
Epoch [2/5], Step [200/1875], Loss: 0.1628
Epoch [2/5], Step [300/1875], Loss: 0.1865
Epoch [2/5], Step [400/1875], Loss: 0.1873
Epoch [2/5], Step [500/1875], Loss: 0.1771
Epoch [2/5], Step [600/1875], Loss: 0.1502
Epoch [2/5], Step [700/1875], Loss: 0.1614
Epoch [2/5], Step [800/1875], Loss: 0.1653
Epoch [2/5], Step [900/1875], Loss: 0.1759
Epoch [2/5], Step [1000/1875], Loss: 0.2007
Epoch [2/5], Step [1100/1875], Loss: 0.1931
Epoch [2/5], Step [1200/1875], Loss: 0.1748
Epoch [2/5], Step [1300/1875], Loss: 0.1879
Epoch [2/5], Step [1400/1875], Loss: 0.1748
Epoch [2/5], Step [1500/1875], Loss: 0.1920
Epoch [2/5], Step [1600/1875], Loss: 0.1794
Epoch [2/5], Step [1700/1875], Loss: 0.1715
Epoch [2/5], Step [1800/1875], Loss: 0.1848
Epoch [3/5], Step [100/1875], Loss: 0.1500
Epoch [3/5], Step [200/1875], Loss: 0.1666
Epoch [3/5], Step [300/1875], Loss: 0.1486
Epoch [3/5], Step [400/1875], Loss: 0.1925
Epoch [3/5], Step [500/1875], Loss: 0.1865
Epoch [3/5], Step [600/1875], Loss: 0.1870
Epoch [3/5], Step [700/1875], Loss: 0.1452
Epoch [3/5], Step [800/1875], Loss: 0.1412
Epoch [3/5], Step [900/1875], Loss: 0.1637
Epoch [3/5], Step [1000/1875], Loss: 0.1664
Epoch [3/5], Step [1100/1875], Loss: 0.1938
Epoch [3/5], Step [1200/1875], Loss: 0.1666
Epoch [3/5], Step [1300/1875], Loss: 0.1645
Epoch [3/5], Step [1400/1875], Loss: 0.1636
Epoch [3/5], Step [1500/1875], Loss: 0.2122
Epoch [3/5], Step [1600/1875], Loss: 0.1579
Epoch [3/5], Step [1700/1875], Loss: 0.1933
Epoch [3/5], Step [1800/1875], Loss: 0.1840
Epoch [4/5], Step [100/1875], Loss: 0.1785
Epoch [4/5], Step [200/1875], Loss: 0.1494
Epoch [4/5], Step [300/1875], Loss: 0.1598
Epoch [4/5], Step [400/1875], Loss: 0.1498
Epoch [4/5], Step [500/1875], Loss: 0.1751
Epoch [4/5], Step [600/1875], Loss: 0.1434
Epoch [4/5], Step [700/1875], Loss: 0.1652
Epoch [4/5], Step [800/1875], Loss: 0.1761
Epoch [4/5], Step [900/1875], Loss: 0.1686
Epoch [4/5], Step [1000/1875], Loss: 0.1746
Epoch [4/5], Step [1100/1875], Loss: 0.1522
Epoch [4/5], Step [1200/1875], Loss: 0.1996
Epoch [4/5], Step [1300/1875], Loss: 0.1791
Epoch [4/5], Step [1400/1875], Loss: 0.1618
Epoch [4/5], Step [1500/1875], Loss: 0.1828
Epoch [4/5], Step [1600/1875], Loss: 0.1656
Epoch [4/5], Step [1700/1875], Loss: 0.1716
Epoch [4/5], Step [1800/1875], Loss: 0.1527
Epoch [5/5], Step [100/1875], Loss: 0.1305
Epoch [5/5], Step [200/1875], Loss: 0.1501
Epoch [5/5], Step [300/1875], Loss: 0.1837
Epoch [5/5], Step [400/1875], Loss: 0.1488
Epoch [5/5], Step [500/1875], Loss: 0.1535
Epoch [5/5], Step [600/1875], Loss: 0.1523
Epoch [5/5], Step [700/1875], Loss: 0.1465
Epoch [5/5], Step [800/1875], Loss: 0.1581
Epoch [5/5], Step [900/1875], Loss: 0.1555
Epoch [5/5], Step [1000/1875], Loss: 0.1875
Epoch [5/5], Step [1100/1875], Loss: 0.1732
Epoch [5/5], Step [1200/1875], Loss: 0.1804
Epoch [5/5], Step [1300/1875], Loss: 0.1622
Epoch [5/5], Step [1400/1875], Loss: 0.1469
Epoch [5/5], Step [1500/1875], Loss: 0.1672
Epoch [5/5], Step [1600/1875], Loss: 0.1433
Epoch [5/5], Step [1700/1875], Loss: 0.1469
Epoch [5/5], Step [1800/1875], Loss: 0.1534
'''

# Function to evaluate the accuracy on test data
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    #now we start evaluation
    model.eval() 
    #no gradients compute-->saves time
    with torch.no_grad(): 
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    
# Call evaluate function and save it
evaluate_model(model, testloader)

####OUTPUT###
'''
Accuracy of the model on the test images: 95.98%
'''
#############

torch.save(model.state_dict(), 'digitClassification_Rohith.pth')
 
