import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNet(nn.Module):
    def __init__(
        self,
        input_size,
        nunit,
        device
    ):
        super(ListNet, self).__init__()
        
        self.unit = nn.Linear(input_size, nunit)
           
        self.activation = lambda x: x
        
        self.device = device
        if device.type != 'cpu':
            self.cuda(device)
    
    def forward(self, X):
        output = self.activation(self.unit(X))
            
        return output
    
    def predict(self, X, k=None):
        if k is None:
            k = X.size(0)
            
        output = self.forward(X).flatten()
        ranking = torch.argsort(
            F.softmax(output, dim=0),
            descending=True
        )
        
        return ranking[:k]

    def predict_qid(self, X_list, k=None):
        rankings = []
        for X in X_list:
            rankings.append(
                self.predict(X, k)
            )

        return rankings


def model_train(
    model,
    optimizer,
    data_loader,
    criterion,
    max_epoch=50,
):
    for epoch in range(max_epoch):
        total_loss = 0
        for iter_num, (X, y) in enumerate(data_loader):
            loss = 0        
            optimizer.zero_grad()

            output = F.softmax(model.forward(X), dim=0).flatten()
            ground_truth = F.softmax(y, dim=0)

            loss += criterion(
                output,
                ground_truth
            )

            loss.backward()
            optimizer.step()
            
            total_loss += loss
            
        if epoch % 10 == 0:
            print(epoch, 'Loss: ', total_loss.item() / len(data_loader))
