import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import src.data_loaders as utilities
import dill
import src.argsparser as argsparser
from src.resnet import ResNet
import numpy as np
import torchviz
from src.utils import Timer
from src.pruning import prune_model , apply_pruning

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
args = argsparser.get_parser().parse_args()
best_prec1 = 0

model_params = [args.model,
                args.dataset,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.experiment_state,
                args.w_bits,
                args.w_bits_per_slice,
                args.wa_stoch_round,
                args.a_bits,
                args.a_bits_per_stream,
                args.save_adc,
                args.adc_prec,
                args.adc_grad_filter,
                args.adc_stoch_round,
                args.adc_static_step,
                args.adc_pos_only,
                args.adc_custom_loss,
                args.shared_adc,
                # args.acm_fixed_bits,
                # args.acm_frac_bits,
                args.slice_init
                ]
quant_add = "_"
for item in model_params:
    quant_add += str(item).replace('.', 'p') + '_'
quant_add = quant_add[:-1]  # Remove the last underscore

run_name = args.run_info + quant_add
print(f"Run info: {args.run_info}")
print(f"Run name: {run_name }")

model_save_dir = args.model_save_dir + run_name + ".th"
logs_save_dir = args.logs_save_dir + run_name + ".txt"

if not os.path.exists("./saved/"):
    os.mkdir("./saved/")

if not os.path.exists(args.model_save_dir):
    os.mkdir(args.model_save_dir)

if not os.path.exists(args.logs_save_dir):
    os.mkdir(args.logs_save_dir)

if not os.path.exists("./saved/hist_csvs/"):
    os.mkdir("./saved/hist_csvs/")

# if not args.evaluate:
Log_Vals = open(logs_save_dir, 'w')

def main():
    start_time = time.time()
    global args, best_prec1
    print(f"Time @ args: {time.time() - start_time}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model according to params
    if args.model == "resnet20":
        num_blocks = [3, 3, 3]
        start_chan = 16
    elif args.model == "resnet18":
        num_blocks = [2, 2, 2, 2]
        start_chan = 64
    else:
        raise NotImplementedError("Not a valid model for current codebase")

    if args.dataset == 'MNIST':
        model = ResNet(num_blocks, 1, args, start_chan).to(device)
    elif args.dataset == 'CIFAR10':
        model = ResNet(num_blocks, 3, args, start_chan).to(device)
    else:
        raise NotImplementedError("Not a valid dataset for current codebase")

    # model = model.half()
    print(f"Time @ model load: {time.time()-start_time}")

    # optionally resume from a checkpoint
    if args.resume:  # Model file must match arch, good luck
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False

    train_loader, val_loader = utilities.get_loaders(dataset=args.dataset, batch_size=args.batch_size, workers=4)
    print(f"Time @ data load: {time.time()-start_time}")

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    # print(model.modules)  # Print all model components/layers
    start_train = time.time()

    print(f"Time @ start: {time.time()-start_time}")

    if (len(args.resume) > 5):
        if args.experiment_state == "inference" or args.save_adc:
            validate(val_loader, model, criterion)
            exit()
            
        if args.experiment_state == "xbar_inference":
            validate(val_loader, model, criterion)
            exit()
        
        if args.experiment_state == "PTQAT": 
            pass       

    # begin epoch training loop
    for epoch in range(0, args.epochs):
        start_epoch = time.time()
        model.train()
        
        if args.experiment_state == "pruning":
            prune_model(model, args.conv_prune_rate, args.linear_prune_rate)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion)
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # Remove pruning reparameterization before saving
        if args.experiment_state == "pruning":
            model = apply_pruning(model)  # <-- Apply before saving

        save_checkpoint({
                'model': model,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(model_save_dir))
        print("Epoch Time: " + str(time.time() - start_epoch))
        Log_Vals.write(str(prec1) + '\n')
    print("Total Time: " + str(time.time() - start_train))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output, loss_add = model(input_var)
        loss = criterion(output, target_var) + loss_add
        
        if args.viz_comp_graph:
            print("Visualizing computational graph, please wait ...")
            torchviz.make_dot(output).render("./saved/torchviz_out", format="png")
            print("Finished visualizing computational graph 1")
            torchviz.make_dot(loss).render("./saved/torchviz_loss", format="png")
            print("Finished visualizing computational graph 2")
            exit()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    Log_Vals.write(str(epoch+1) + ', ' + str(losses.avg) + ', ' + str(top1.avg) + ', ')


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    end = time.time()
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        last_finished_batch = args.skip_to_batch
        if i < last_finished_batch:
            continue

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

        # compute output
            output, loss_add = model(input_var)

            loss = criterion(output, target_var) + loss_add

            output = output.float()
            loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.print_batch_info:
            string = f"Batch {i+1} Prec@1 = {prec1:.2f}%, Avg = {top1.avg:.2f}%"
            print(string)
            with open(args.batch_log, 'a') as log_file:
                log_file.write(string)


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save the training model"""
    torch.save(state, filename, pickle_module=dill)


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
