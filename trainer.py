import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
TRAIN_MODEL = 'FA'
epochs = 10


# download/load train dataset
train_loader = DataLoader(
    datasets.MNIST(
        './data', train=True, download=True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),(0.3081,)
            )
        ])
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,    
)

# download/load train dataset
test_loader = DataLoader(
    datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),(0.3081,)
            )        
        ]),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


from models.FA import *
# load ff fa model
# model = LinearFANetwork()

model = LinearFANetwork(
    in_features=784, 
    num_layers=2, 
    num_hidden_list=[1000, 10]
).to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True,
)

loss_crossentropy = torch.nn.CrossEntropyLoss()

for epoch in tqdm(range(epochs)):
    for idx_batch, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(BATCH_SIZE, -1)

        # autograd vars
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs.to(device))

        loss = loss_crossentropy(outputs, targets.to(device))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # if (...):
        #     logging...


