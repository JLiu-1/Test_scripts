import numpy as np 


from heat import *
import os
import argparse
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchsummary import summary
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, l1_decay=0):
    """
        Run one train epoch
    """
   
    losses = AverageMeter()
    
    
    # switch to train mode
    model.train()

   
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
       
        if args.gpu:
            target = target.cuda()
            input_var = input.cuda()
        target_var = target
        
            
        # compute output

        output = model(input_var)
       
       
        loss = criterion(output, target_var)
        if l1_decay>0:
            for param in model.parameters():
                loss+=l1_decay*torch.sum((torch.abs(param)))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        
        losses.update(loss.item(), input.size(0))
        
        print([epoch,i,losses.avg])
        

        # measure elapsed time
       
        
            #
    return losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
   
    losses = AverageMeter()
    

    # switch to evaluate mode
    model.eval()

   
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu:
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            #prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            

            # measure elapsed time
            
            

    print('Val * Loss@1 {losses.avg:.3f}'
          .format(losses=losses))

    return losses.avg

parser = argparse.ArgumentParser()
#parser.add_argument('--dropout','-d',type=float,default=0)
parser.add_argument('--learningrate','-l',type=float,default=1e-3)

#parser.add_argument('--hidden_dims','-k',type=int,default=10)
parser.add_argument('--batchsize','-b',type=int,default=64)
parser.add_argument('--epoch','-e',type=int,default=100)
parser.add_argument('--actv','-a',type=str,default='no')
#parser.add_argument('--field','-f',type=str,default='baryon_density')
parser.add_argument('--norm_min','-m',type=float,default=-1)
parser.add_argument('--normalize','-n',type=float,default=0)
#parser.add_argument('--conv','-c',type=int,default=0)
#parser.add_argument('--noise','-n',type=float,default=0)
#parser.add_argument('--trainlog','-tl',type=str,default='train.log')
#parser.add_argument('--vallog','-vl',type=str,default='val.log')
parser.add_argument('--gpu','-g',type=int,default=1)
parser.add_argument('--double','-d',type=int,default=0)
parser.add_argument('--save','-s',type=str,default="ckpts_heat")
parser.add_argument('--save_interval','-i',type=int,default=10)
#parser.add_argument('--ratio','-r',type=int,default=1)
#parser.add_argument('--random','-rd',type=int,default=0)
parser.add_argument('--l2decay','-l2d',type=float,default=0)
parser.add_argument('--l1decay','-l1d',type=float,default=0)
args = parser.parse_args()
actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
actv=actv_dict[args.actv]
bs=args.batchsize
lr=args.learningrate
#field=args.field
#ratio=args.ratio
#k=args.hidden_dims

max_epoch=args.epoch
interval=args.save_interval
path="/home/jliu447/lossycompression/Heat"
#val_path="/home/jliu447/lossycompression/NYX"
#maximum={"baryon_density":5.06394195556640625,"temperature":6.6796627044677734375,"dark_matter_density":4.1392154693603515625}
#minimum={"baryon_density":-1.306397557258605957,"temperature":2.7645518779754638672,"dark_matter_density":-10}



#torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
#scheduler.step()


layer=nn.Conv2d(1,1,3,padding=1,bias=False)

#ini=torch.Tensor([[0.001,0.249,0.001,0.249,-0.001,0.249,-0.001,0.249,0.001]])
#lin.weight=torch.nn.Parameter(ini)
model=nn.Sequential(layer,actv())
if args.double:
    model=model.double()

optimizer=torch.optim.SGD(model.parameters(), lr=lr,weight_decay=args.l2decay)
#optimizer=torch.optim.SGD(model.parameters(), lr=lr)

criterion = nn.MSELoss()

if args.gpu:
    model=model.cuda()
    criterion=criterion.cuda()

try:
    summary(model,(1,256,256,))
except:
    print("Failed to summary.")

start=21000
end=29000
gmax=1000
gmin=0
sizex=200
sizey=200


if args.normalize:
    if args.double:
        train_loader = DataLoader(
            Heat_Full_Double(path,start,end,sizex,sizey,global_max=gmax,global_min=gmin,norm_min=args.norm_min),
            batch_size=bs, shuffle=True,
            num_workers=0)
    else:
        train_loader = DataLoader(
            Heat_Full(path,start,end,sizex,sizey,global_max=gmax,global_min=gmin,norm_min=args.norm_min),
            batch_size=bs, shuffle=True,
            num_workers=0)
else:
    if args.double:
        train_loader = DataLoader(
            Heat_Full_Double(path,start,end,sizex,sizey,global_max=None,global_min=None,norm_min=args.norm_min),
            batch_size=bs, shuffle=True,
            num_workers=0)
    else:
        train_loader = DataLoader(
            Heat_Full(path,start,end,sizex,sizey,global_max=None,global_min=None,norm_min=args.norm_min),
            batch_size=bs, shuffle=True,
            num_workers=0)



#print(y[:100])

if not os.path.exists(args.save):
    os.makedirs(args.save)


for epoch in range(max_epoch):

        # train for one epoch
   
    loss_train=train(train_loader, model, criterion, optimizer, epoch, args.l1decay)
    print('Train * Loss@1 {loss_train:.3f}'
          .format(loss_train=loss_train))
    
    #lr_scheduler.step()
        

        # evaluate on validation set
    #loss1 = validate(val_loader, model, criterion)
   
        # remember best prec@1 and save checkpoint
    if epoch % interval==0 or epoch==max_epoch-1:
        torch.save({"state_dict":model.state_dict(),"epoch":epoch,"lr":lr},os.path.join(args.save,"ckpt_%d.pth" % epoch))







#validate(val_loader,model,criterion)

