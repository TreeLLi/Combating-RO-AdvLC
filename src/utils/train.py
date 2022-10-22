import time, os

import torch as tc
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import torch.utils.data.distributed as dd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import grad

from src.data.factory import fetch_dataset, DATASETS
from src.model.factory import *
from src.utils.helper import *
from src.utils.evaluate import validate
from src.utils.printer import sprint, dprint
from src.utils.adversary import pgd
from src.utils.swa import moving_average, bn_update

def train(args):
    start_epoch = 0
    best_acc1, best_pgd, best_fgsm = 0, 0, 0
    checkpoint = None
    
    if args.resume is not None:
        resume_file = args.path('trained', "{}/{}_end".format(args.logbook, args.resume))
        if os.path.isfile(resume_file):
            checkpoint = tc.load(resume_file, map_location='cpu')
            best_acc1 = checkpoint['best_acc1']
            best_fgsm = checkpoint['best_fgsm']
            best_pgd = checkpoint['best_pgd']
            start_epoch = checkpoint['epoch']
        else:
            raise Exception("Resume point not exists: {}".format(args.resume))
        checkpoint = (args.resume, checkpoint)
        
    fargs = args.func_arguments(fetch_model, ARCHS, postfix='arch')
    if checkpoint is not None:
        fargs['checkpoint'] = checkpoint
    model = fetch_model(**fargs).to(args.device)
    if args.parallel:
        if args.using_cpu():
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[args.rank], output_device=args.rank)

    if args.swa is not None:
        if args.resume is None or start_epoch <= args.swa:
            swa_model = fetch_model(**fargs).to(args.device)
            swa_best_acc = 0.0
            swa_best_fgm = 0.0
            swa_best_pgd = 0.0
            args.swa_n = 0
        else:
            swa_ckp = args.path('trained', "{}/{}_swa_end".format(args.logbook, args.resume))
            swa_ckp = tc.load(swa_ckp, map_location='cpu')
            fargs['checkpoint'] = (args.resume, swa_ckp)
            swa_model = fetch_model(**fargs).to(args.device)
            swa_best_acc = swa_ckp['best_acc']
            swa_best_fgm = swa_ckp['best_fgm']
            swa_best_pgd = swa_ckp['best_pgd']
            args.swa_n = swa_ckp['num']
    else:
        swa_model = None
        
    criterion = nn.CrossEntropyLoss()

    fargs = args.func_arguments(fetch_optimizer, OPTIMS, checkpoint=checkpoint)
    optimizer = fetch_optimizer(params=model.parameters(), **fargs)
    
    fargs = args.func_arguments(fetch_dataset, DATASETS, postfix='data')
    train_set = fetch_dataset(train=True, **fargs)
    train_sampler = dd.DistributedSampler(train_set) if args.parallel else None
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              drop_last=True)

    val_set = fetch_dataset(train=False, split=args.world_size, **fargs)
    if args.world_size > 1:
        total_samples = sum([len(vs) for vs in val_set])
        val_set = val_set[args.rank]
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    dprint('adversary', **{k:getattr(args, k, None)
                           for k in ['eps', 'eps_step', 'max_iter', 'random_init', 'eval_iter']})
    dprint('data loader', batch_size=args.batch_size, num_workers=args.num_workers)
    sprint("=> Start training!", split=True)

    for epoch in range(start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr, args.annealing)

        update(train_loader, model, criterion, optimizer, epoch, args, swa_model)
        
        model.eval()
        acc1, ig, fgsm, pgd = validate(val_loader, model, criterion, args)
        
        if args.rank is not None and args.rank != 0: continue
        # execute only on the main process
        
        best_acc1 = max(acc1, best_acc1)
        best_fgsm = max(fgsm, best_fgsm)
        best_pgd = max(pgd, best_pgd)

        print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(best_acc1, best_fgsm, best_pgd))
        
        if args.logging:
            logger = args.logger
            acc_info = '{:3.2f} E: {} IG: {:.2e} FGSM: {:3.2f} PGD: {:3.2f}'.format(acc1, epoch+1, ig, fgsm, pgd)
            logger.update('checkpoint', end=acc_info)
            state_dict = model.module.state_dict() if args.parallel else model.state_dict()
            state = {
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'best_pgd': best_pgd,
                'best_fgsm' : best_fgsm,
                'optimizer' : optimizer.state_dict(),
            }

            if acc1 >= best_acc1:
                logger.update('checkpoint', acc=acc_info, save=True)

            lid = args.log_id
            fname = "{}/{}".format(args.logbook, lid)
            ck_path = args.path('trained', fname+"_end")
            tc.save(state, ck_path)

            if acc1 >= best_acc1:
                shutil.copyfile(ck_path, args.path('trained', fname+'_acc'))

            if pgd >= best_pgd:
                logger.update('checkpoint', pgd=acc_info, save=True)
                shutil.copyfile(ck_path, args.path('trained', fname+'_pgd'))

            if args.swa is not None and args.swa_start <= epoch:
                print(" *  averaging the model")
                swa_model.eval()
                swa_acc, swa_ig, swa_fgm, swa_pgd = validate(val_loader, swa_model, criterion, args)
                
                swa_best_acc = max(swa_acc, swa_best_acc)
                swa_best_fgm = max(swa_fgm, swa_best_fgm)
                swa_best_pgd = max(swa_pgd, swa_best_pgd)
                
                print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(swa_best_acc,
                                                                              swa_best_fgm,
                                                                              swa_best_pgd))
                
                state = {'state_dict' : swa_model.state_dict(),
                         'num': args.swa_n,
                         'best_acc' : swa_best_acc,
                         'best_pgd' : swa_best_pgd,
                         'best_fgm' : swa_best_fgm}

                ck_path = args.path('trained', fname+"_swa_end")
                tc.save(state, ck_path)

                if swa_pgd >= swa_best_pgd:
                    shutil.copyfile(ck_path, args.path('trained', fname+'_swa_pgd'))

def update(loader, model, criterion, optimizer, epoch, args, swa_model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    igs = AverageMeter('IG', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    meters = [batch_time, losses, igs, top1, top5]
    progress = ProgressMeter(len(loader), meters, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    niter = len(loader)
    eps, step = args.eps, args.eps_step
    for i, (img, tgt) in enumerate(loader, 1):
        img = img.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)        
        prt = tc.zeros_like(img)
        if args.random_init: prt.uniform_(-eps, eps)
        
        batch_size = len(img)

        cle = tc.clamp(img+prt, 0, 1).requires_grad_(True)
        lgt_cle = model(cle)
        loss = criterion(lgt_cle, tgt)
        
        ig = grad(loss, cle, retain_graph=True, create_graph=False)[0]
        ig_norm = tc.norm(ig, p=1)
        adv, prt = pgd(cle, tgt, model, criterion, eps, step, args.max_iter, ig)

        lgt_adv = model(adv)
        opt = lgt_adv
        loss = criterion(lgt_adv, tgt)
        
        if args.reg_top == 1:
            reg = tc.norm(lgt_cle-lgt_adv, p=1) / batch_size
        else:
            diff = tc.abs(lgt_cle-lgt_adv).view(-1)
            topk = int(tc.numel(diff)*args.reg_top)
            topk, _ = tc.topk(diff, topk, largest=True)
            reg = tc.sum(topk) / batch_size

        lam = args.reg_lam*(epoch+1)/args.epochs if args.reg_warmup else args.reg_lam
        loss += lam * reg
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(opt, tgt, topk=(1, 5))
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)
        losses.update(loss.item(), batch_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        igs.update(ig_norm, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i == 1 or i % args.log_pbtc == 0:
            progress.display(i)
        
        if args.rank != 0: continue
        
        if args.swa is not None and args.swa_start <= epoch and i % args.swa_freq == 0:
            if isinstance(args.swa_decay, str):
                moving_average(swa_model, model, 1.0 / (args.swa_n + 1))
                args.swa_n += 1
            else:
                moving_average(swa_model, model, args.swa_decay)
            bn_update(loader, swa_model)
        
        end = time.time()
        

def adjust_learning_rate(optimizer, epoch, lr, annealing):
    decay = 0
    for a in annealing:
        if epoch < int(a): break
        else: decay += 1
    
    lr *= 0.1 ** decay

    params = optimizer.param_groups
    if lr != params[0]['lr']:
        sprint("Learning rate now is {:.0e}".format(lr))
    
    for param in params: param['lr'] = lr

