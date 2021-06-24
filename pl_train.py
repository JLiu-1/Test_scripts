from nyx import NYX
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import *
from pytorch_lightning import Trainer

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dropout','-d',type=float,default=0)
    parser.add_argument('--learningrate','-l',type=float,default=1e-3)

    #parser.add_argument('--hidden_dims','-k',type=int,default=10)
    parser.add_argument('--batchsize','-b',type=int,default=2048)
    parser.add_argument('--epoch','-e',type=int,default=100)
    parser.add_argument('--actv','-a',type=str,default='tanh')
    parser.add_argument('--field','-f',type=str,default='baryon_density')
    parser.add_argument('--norm_min','-m',type=float,default=-1)
    #parser.add_argument('--noise','-n',type=float,default=0)
    #parser.add_argument('--gpu','-g',type=int,default=1)
    parser.add_argument('--save','-s',type=str,default="ckpts")
    parser.add_argument('--save_interval','-i',type=int,default=10)
    parser.add_argument('--ratio','-r',type=int,default=10)
    args = parser.parse_args()

    gpu_list=os.getenv("CUDA_VISIBLE_DEVICES")
    gpu_list=list(eval(gpu_list))

    actv=args.actv
    bs=args.batchsize
    lr=args.learningrate
    field=args.field
    ratio=args.ratio
    #k=args.hidden_dims

    max_epoch=args.epoch
    interval=args.save_interval
    path="/home/jliu447/lossycompression/NYX"
    #val_path="/home/jliu447/lossycompression/NYX"
    maximum={"baryon_density":5.06394195556640625,"temperature":6.6796627044677734375,"dark_matter_density":4.1392154693603515625}
    minimum={"baryon_density":-1.306397557258605957,"temperature":2.7645518779754638672,"dark_matter_density":-10}



#torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
#scheduler.step()

    model=NNpredictor(actv)






    train_loader = DataLoader(
        NYX(path,field,0,3,ratio=ratio,log=1,global_max=maximum[field],global_min=minimum[field],norm_min=args.norm_min),
        batch_size=bs, shuffle=True,
        num_workers=0)
    '''
    val_loader = DataLoader(
            NYX(path,field,3,4,ratio=ratio,log=1,global_max=maximum[field],global_min=minimum[field],norm_min=args.norm_min),
           batch_size=32768, shuffle=False,
            num_workers=0)
'''
    #print(y[:100])

    if not os.path.exists(args.save):
        os.makedirs(args.save)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save,
        save_top_k=-1,
        verbose=True,
        #save_last=True,
        #monitor='loss',
        #mode='min',
        
        period=save_interval
    )

    runner = Trainer(min_epochs=1,
                 checkpoint_callback=True,
                 callbacks=checkpoint_callback,
                 accelerator=args.accelerator,
                 **config['trainer_params'])

    print(f"======= Training =======")
    runner.fit(experiment)
    runner.save_checkpoint(args.save+"/last.ckpt")





#validate(val_loader,model,criterion)

