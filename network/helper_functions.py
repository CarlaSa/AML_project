import torch
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix



def get_multilabel_confusion_matrix(model, dataloader):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ylist =torch.zeros(0, device='cpu')
    outlist =torch.zeros(0, device='cpu')
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            x = x.float()
            y = y.float()
            out = model(x)
            ylist =torch.cat([ylist,y.cpu()])
            out = out >0.3
            outlist=torch.cat([outlist,out.cpu()])
    return multilabel_confusion_matrix(ylist, outlist)


def get_confusion_matrix(model, dataloader):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ylist =torch.zeros(0, device='cpu')
    outlist =torch.zeros(0, device='cpu')
    with torch.no_grad():
        for x,y in dataloader_val:
            x = x.to(device)
            y = y.to(device)
            x = x.float()
            y = y.float().argmax(dim = 1)
            out = model(x)
            ylist =torch.cat([ylist,y.cpu()])
            out = torch.argmax(out, dim = 1)
            outlist=torch.cat([outlist,out.cpu()])
    return confusion_matrix(ylist, outlist)

