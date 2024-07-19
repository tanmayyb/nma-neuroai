import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
TRAIN_MODEL = 'FA'
epochs = 100

# get dataloader from noteboook?


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


from models import fa
# load ff fa model
model = fa.FA()

optimizer = torch.optim.SGD(
    #...
)

loss_crossentropy = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for idx_batch, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(BATCH_SIZE, -1)

        # autograd vars
        inputs, targets = Variable(inputs), Variable(targets)


        outputs = model(inputs.to(device))

        loss  = model(outputs, targets.to(device))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # if (...):
        #     logging...


