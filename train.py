from __future__ import print_function
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc
import dataset
from utils import *
from image import correct_yolo_boxes
from cfg import parse_cfg
from darknet import Darknet
import argparse
import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

FLAGS = None
unparsed = None
device = None

# global Training settings
use_cuda      = None
eps           = 1e-5
keep_backup   = 200
save_interval = 1  # epoches
test_interval = 200  # epoches
dot_interval  = 10  # batches

# Test parameters
evaluate = False
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# no test evalulation
# eval = True
init_eval = False

### some global variable we can change during training
condition = False
adaptation = 0
layerwise = 0
learning_rate = 1e-3
max_epochs = 0
classify_loss_weight = 1

def main():
    datacfg    = FLAGS.data
    cfgfile    = FLAGS.config
    weightfile = FLAGS.weights
    # eval    = FLAGS.eval
    continuetrain = FLAGS.continuetrain
    adaptation = FLAGS.adaptation
    layerwise = FLAGS.layerwise
    max_epochs = FLAGS.epoch
    # condition = FLAGS.condition

    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
    globals()["trainlist"]     = data_options['train']
    globals()["trainlist_aug"] = data_options['train_aug']
    globals()["testlist"]      = data_options['valid']
    globals()["classname"]     = data_options['names']
    globals()["backupdir"]     = data_options['backup']
    globals()["gpus"] = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"]         = len(gpus.split(','))
    globals()["num_workers"]   = int(data_options['num_workers'])
    globals()["batch_size"]    = int(net_options['batch'])
    globals()["max_batches"]   = int(net_options['max_batches'])
    globals()["burn_in"]       = int(net_options['burn_in'])
    # globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [int(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]

    learning_rate = float(net_options['learning_rate'])
    try:
        globals()["backupdir"] = data_options['backup']
    except:
        globals()["backupdir"] = 'backup'

    if not os.path.exists(backupdir):
        os.mkdir(backupdir)

    try:
        globals()["logfile"] = data_options['logfile']
    except:
        globals()["logfile"] = 'backup/logfile.txt'

    try:
        globals()["condition"] = bool(net_options['condition'])
    except:
        globals()["condition"] = False

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)

    model.print_network()
    nsamples = file_lines(trainlist)
    #initialize the model
    if FLAGS.reset:
        model.seen = 0
        init_epoch = 0
    else:
        init_epoch = model.seen//nsamples
    iterates = 0
    savelog('# Defaults configurations:')
    savelog('# Trainlist %s, Testlist %s' % (trainlist, testlist))
    savelog('# Cfg file: %s' % (cfgfile))
    savelog('# Max_epoch: %d, batchsize: %d, burn_in %d , Learning rate: %e' % (
        max_epochs, batch_size, burn_in, learning_rate))
    savelog('# Image size (width and height): %d x %d' % (model.width, model.height))
    savelog('# Step: %s' % (steps))
    if condition:
        savelog('# Training with conditioning_net = %d ' % (condition))
    if adaptation>0:
        savelog('# Training Adaptation the first %d layers' % (adaptation))
    if layerwise > 0:
        savelog('# Training Layerwise every %d layers ' % (layerwise))


    if weightfile is not None:
        model.load_weights(weightfile)
        savelog('# Load weight file from %s' % (weightfile))
        continuetrain = None

    if continuetrain is not None:
        checkpoint = torch.load(continuetrain)
        model.load_state_dict(checkpoint['state_dict'])
        try:
            init_epoch = int(continuetrain.split('.')[0][-2:])
        except:
            logging('Warning!!! Continuetrain file must has at least 2 number at the end indicating last epoch')
        iterates = init_epoch*(nsamples/batch_size)
        savelog('# Training continue from model %s' % continuetrain)
        savelog('# Training starting from %d epoch with %d iterates ' %(init_epoch,iterates))


    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).to(device)
            logging('Use CUDA train on %s GPUs' % (gpus))
        else:
            model = model.to(device)
            logging('Use CUDA train only 1 GPU')

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    global optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)
    savelog('# Optimizer: SGD with learning rate: %f momentum %f weight_decay %f' % (
    learning_rate / batch_size, momentum, decay * batch_size))
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    if continuetrain is not None:
        checkpoint = torch.load(continuetrain)
        optimizer.load_state_dict(checkpoint['optimizer'])

    if adaptation > 0:
       savelog('# Training Segment Adaptation: ')
       freeze_weight_adaptation(adaptation)

    # global train_dataset, valid_dataset
    global train_dataset, valid_dataset

    cur_loss, best_loss = 0.0, 10000.0
    best_pre, cur_pre, cur_rec, best_rec = 0.0, 0.0, 0.0, 0.0
    lr_time, loss_time, rec_time = 0, 0, 0

    try:

        savelog("# Training for ({:d},{:d})".format(init_epoch + 1, max_epochs))
        for epoch in range(init_epoch+1, max_epochs+1):
            ### Split trainsampler and validsampler from the trainset.
            train_sampler, valid_sampler = get_train_valid_sampler()
            if condition:
                iterates, cur_loss = train_conditioning(epoch,iterates,train_sampler)
            else:
                ### This is for layerwise and normally training.
                if layerwise > 0:
                    layerwise = update_weight_layerwise(epoch,layerwise)

                iterates,cur_loss = train(epoch,iterates,train_sampler,adaptation,layerwise)

            ### validate
            if 111 < 100:
                cur_fscore, cur_pre, cur_rec = PR_Valid(epoch, valid_sampler)

            #savelog("%d trainloss: %f fscore: %.4f precision: %.4f recall: %.4f" % (epoch, cur_loss,cur_fscore,cur_rec,cur_rec))
            savemodel(epoch, cfgfile, cur_pre, True)

            ### This is important procedure we invent for monitor training procedure, reduce waiting time.
            ### The idea is that if the network doesn't learn (loss increase), and valid precision too, --> derease lr.
            ### changing lr by precision (check fast training procedure)

            if ((adaptation > 0) or (layerwise > 0)):
                adaptation, layerwise, best_loss, loss_time, best_rec = check_update_require_grad(adaptation, layerwise, cur_loss, best_loss, loss_time, cur_rec, best_rec)
            else:
                best_loss, loss_time, best_rec, rec_time, learning_rate, lr_time = check_change_lr(
                    cur_loss, best_loss, loss_time, cur_rec, best_rec, rec_time, learning_rate, lr_time)
                # learning_rate = change_lr_by_epoch(epoch, learning_rate)
            logging('-' * 90)

    except KeyboardInterrupt:
        logging('='*80)
        logging('Exiting from training by interrupt')



def check_update_require_grad(adaptation, layerwise, cur_loss, best_loss, loss_time, cur_rec, best_rec):
    if cur_loss > best_loss:
        ### loss increases (unnormal)
        loss_time += 1
    else:
        best_loss = cur_loss
        if loss_time > 0:
            if cur_rec > best_rec:
                loss_time = 0       ### reset loss time.
                best_rec = cur_rec
                savelog('# Fine-tuning normally: ')
                layers_update = []
                for i, (name, para) in enumerate(model.named_parameters()):
                    layers_update.append(i)
                    para.requires_grad = True
                    # logging('%d name: %s grad: %s ' % (i, name, para.requires_grad))
                savelog('# Layers to update weights :' % (layers_update))
                adaptation = 0
                layerwise = 0
    if cur_rec > best_rec:
        best_rec = cur_rec
    return adaptation,layerwise,best_loss,loss_time,best_rec



def check_change_lr(cur_loss, best_loss, loss_time, cur_rec, best_rec, rec_time, lr, lr_time):
    if cur_loss > best_loss:
        ### loss increases (unnormal)
        loss_time += 1
    else:
        best_loss = cur_loss
        if loss_time > rec_time:
            if cur_rec > best_rec:
                best_rec = cur_rec
                rec_time +=1
                ### this epoch is good, should change learning rate if possible
                if lr_time < 2:
                    lr_time += 1
                    lr = update_learningrate(lr)
                    savelog('# Changing learning rate = %e ' % (lr))
                else:
                    savelog('# Network stop working because changed lr 2 times')
                    sys.exit()
    if cur_rec > best_rec:
        best_rec = cur_rec
    return best_loss,loss_time,best_rec,rec_time, lr,lr_time


def change_lr_by_epoch(epoch, lr):
    if epoch in steps:
        lr = update_learningrate(lr)
        savelog('# Decrease lr by epoch = %e ' % (lr))
    return lr


def freeze_weight_conditioning():
    list_index = [12, 13, 14, 15, 178, 179, 180, 181, 205, 206, 207, 208, 232, 233, 234, 235, 238, 239]
    layers_update = []
    layers_freeze = []
    for i, (name, para) in enumerate(model.named_parameters()):
        # logging(i, ' ', name, ' grad: ', para.requires_grad)
        # logging(para)
        if i in list_index:
            layers_freeze.append(i)
            para.requires_grad = False
        else:
            layers_update.append(i)
    savelog('# Update weights :' % (layers_update))
    # for i, (name, para) in enumerate(model.named_parameters()):
    #     logging(i, ' ', name, ' grad: ', para.requires_grad)

    savelog('# Freeze layers of Conditioning:')
    for i, (name, para) in enumerate(model.named_parameters()):
        savelog('# %d name: %s grad: %s ' % (i, name, para.requires_grad))


def update_training_adaptation_layerwise(epoch, adaptation, layerwise, train_loss, best_loss,cur_precision,best_precision,lr,lr_time):
    if train_loss > best_loss:
        if ((adaptation > 0) or (layerwise > 0)):
            savelog('# Fine-tuning normally: ')
            layers_update = []
            for i, (name, para) in enumerate(model.named_parameters()):
                layers_update.append(i)
                para.requires_grad = True
            savelog('# Update weights :' % (layers_update))
            adaptation = 0
            layerwise = 0
        if cur_precision < best_precision:
            if lr_time < 2:
                lr_time += 1
                lr = update_learningrate(lr)
                savelog('# Validation Precision decreases at epoch %d, decay learning rate = %e ' % (epoch,lr))
            else:
                savelog('# Network stop working because changed lr 2 times')
                sys.exit()
        else:
            best_precision = cur_precision
    else:
        best_loss = train_loss
    return adaptation, layerwise, best_loss, best_precision, lr, lr_time


def update_weight_layerwise(epoch, layerwise):
    layers_update, layers_freeze = [], []
    count_layers = 0
    for i, (name, para) in enumerate(model.named_parameters()):
        # this increase gradually every 1 convolution layer (3 bz conv, batchnorm, bias)
        if i >= (epoch * layerwise * 3):
            layers_freeze.append(i)
            para.requires_grad = False #sembrerebbe semplicemente questo a ghiacciare i layers
            count_layers += 1
        else:
            layers_update.append(i)
            para.requires_grad = True
    savelog('# Update weights: %s '% (layers_update))
    if count_layers == 0:
        savelog('# Update all weights, finetune normally :')
        layerwise = 0

    return layerwise


def freeze_weight_adaptation(adaptation):
    layers_update = []
    layers_freeze = []
    for i, (name, para) in enumerate(model.named_parameters()):
        # logging(i, ' ', name, ' grad: ', para.requires_grad)
        # this increase gradually every 1 convolution layer (3: bz conv, batchnorm, bias)
        if i >= adaptation * 3:
            layers_freeze.append(i)
            para.requires_grad = False
        else:
            layers_update.append(i)
            para.requires_grad = True
    savelog('# Update weights : %s ' % layers_update)
    savelog('# Training Segment Adaptation:')
    for i, (name, para) in enumerate(model.named_parameters()):
        savelog('# %d name: %s grad: %s ' % (i,name,para.requires_grad))

def get_train_valid_sampler():
    global train_dataset, valid_dataset
    init_width, init_height = model.module.width, model.module.height
    train_dataset = dataset.listDataset(trainlist, shape=(init_width, init_height), shuffle=True,
                                       transform=transforms.Compose([transforms.ToTensor()]),
                                       train=True, seen=model.module.seen, batch_size=batch_size,
                                       num_workers=num_workers)

    valid_dataset = dataset.listDataset(trainlist, shape=(init_width, init_height), shuffle=True,
                                        transform=transforms.Compose([transforms.ToTensor()]),
                                        train=False, seen=model.module.seen, batch_size=batch_size,
                                        num_workers=num_workers)

    valid_size = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_learningrate(lr):
    new_lr = lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr/batch_size
    return new_lr


def curmodel():
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    return cur_model


def get_gradient():
    count_grad,no_grad = 0, 0
    ls_grad = []
    for i in range(len(model.module.models)):
        try:
            if int(len(model.module.models[i])) > 1:
                grad = (model.module.models[i][0].weight.grad).mean(dim=3).mean(dim=2).mean(dim=1)
                grad = ((grad**2).sum()).sqrt()
                count_grad +=1
                ls_grad.append(grad)
        except:
            no_grad += 1
    return ls_grad


def train(epoch,iterates,train_sampler,adaptation,layerwise):
    #Remind to comment lines 440-444 if you wanna train the model without augmentation
    #Also, pass trainlist instead of concat_dataset
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_aug_dataset = dataset.listDataset(trainlist_aug, shape=(640,512), shuffle=True,
                                       transform=transforms.Compose([transforms.ToTensor()]),
                                       train=True, seen=model.module.seen, batch_size=batch_size,
                                       num_workers=num_workers)
    concat_dataset = torch.utils.data.ConcatDataset([train_dataset,train_aug_dataset])
    train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size=batch_size,
                                                **kwargs)
    print(len(concat_dataset))
    print(len(train_loader))
    lr = get_lr()
    logging('[%03d] training processed %d samples, lr %e' % (epoch, iterates, lr))

    model.train()
    train_loss_epoch = 0.0
    train_count = 0

    # grad_accumulate = np.zeros(72)

    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        iterates += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        org_loss = []
        total_batch_loss = 0.0
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            total_batch_loss += ol.item()
            org_loss.append(ol)

        total_batch_loss = total_batch_loss / data.size(0)

        sum(org_loss).backward()
        # if layerwise == 0 and adaptation == 0:
        #     grad_temp = get_gradient()
        #     for iii,element_grad in enumerate(grad_temp):
        #         grad_accumulate[iii] += element_grad

        nn.utils.clip_grad_norm_(model.parameters(), 10000)

        optimizer.step()

        train_count += 1

        train_loss_epoch += total_batch_loss

        del data, target
        gc.collect()

    # with open(logfile,"a+") as myfile:
    #     myfile.writelines("%d trainloss: %f " % (epoch, train_loss_epoch/train_count))
    # # if layerwise ==0 and adaptation ==0:
    # #     with open('backup/gradients.txt',"a+") as mynewfile:
    # #         for element in grad_accumulate:
    # #             mynewfile.writelines('%.5f ' % element)
    # #         mynewfile.writelines('\n')
    #
    # logging("epoch(s): %d train_loss: %f " % (epoch, train_loss_epoch/train_count))

    return iterates,train_loss_epoch / train_count

### this function for training our task-conditioning network
def train_conditioning(epoch,iterates,train_sampler):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    lr = get_lr()
    logging('[%03d] training processed %d samples, lr %e' % (epoch, iterates, lr))

    model.train()
    train_loss_epoch, classify_loss_epoch, accuracy_epoch, = 0.0, 0.0, 0.0
    train_count = 0
    for batch_idx, (data, (target,cls_target)) in enumerate(tqdm.tqdm(train_loader)):
        iterates += 1
        data, target,cls_target = data.to(device), target.to(device), cls_target.to(device)

        optimizer.zero_grad()

        output,cls_output = model(data)

        org_loss = []
        total_batch_loss = 0.0
        count_y_layers = 0
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            total_batch_loss += ol.item()
            count_y_layers += 1
            org_loss.append(ol)
        total_batch_loss /= count_y_layers
        total_batch_loss = total_batch_loss / data.size(0)

        cls_target = cls_target.float().view_as(cls_output)
        sm = nn.Sigmoid()
        bce = nn.BCELoss()
        classify_loss = bce(sm(cls_output), cls_target)
        org_loss.append(classify_loss*classify_loss_weight)
        classify_loss_epoch += classify_loss.item() / data.size(0)

        sum(org_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optimizer.step()

        accuracy = (torch.as_tensor((sm(cls_output) >= 0.5),dtype=torch.float) == cls_target.cpu()).float().mean()
        accuracy_epoch += accuracy.item() / data.size(0)

        train_count += 1

        train_loss_epoch += total_batch_loss

        del data, target, cls_target
        gc.collect()

    # with open(logfile,"a+") as myfile:
    #     myfile.writelines("%d yloss: %.4f closs: %.4f acc: %.4f " % (epoch, train_loss_epoch/train_count,classify_loss_epoch/train_count,accuracy_epoch/train_count))
    #
    # logging("%d trainloss: %f classloss: %f acc: %f " % (
    # epoch, train_loss_epoch / train_count, classify_loss_epoch / train_count, accuracy_epoch / train_count))

    return iterates,train_loss_epoch / train_count


def PR_Valid(epoch,valid_sampler):
    '''
    This function is the same with precision and recall function, but validate on the valid set (not on test set).
    :param epoch:
    :param valid_sampler:
    :return:
    '''
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    cur_model = curmodel()

    valid_batchsize = 1
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batchsize, sampler=valid_sampler, **kwargs)

    num_classes = cur_model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    if cur_model.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(cur_model.width, cur_model.height)
    with torch.no_grad():
        for data, target, org_w, org_h in tqdm.tqdm(valid_loader):
            data = data.to(device)
            if condition:
                output, cls_output = model(data)
            else:
                output = model(data)
            all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes, use_cuda=use_cuda)

            for k in range(len(all_boxes)):
                boxes = all_boxes[k]
                correct_yolo_boxes(boxes, org_w[k], org_h[k], cur_model.width, cur_model.height)
                boxes = np.array(nms(boxes, nms_thresh))

                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)
                if num_pred == 0:
                    continue

                proposals += int((boxes[:,4]>conf_thresh).sum())
                for i in range(num_gts):
                    gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred,1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    # savelog("[%03d] vfscore: %.4f vpre: %.4f vrec: %.4f \n" % (epoch, fscore, precision, recall))
    # with open(logfile,"a+") as myfile:
    #     myfile.writelines("vfscore: %.4f vpre: %.4f vrec: %.4f \n"%(fscore,precision,recall))
    return fscore, precision, recall


def savemodel(epoch, netname,best_pre, savecondition=False):
    cur_model = curmodel()
    state = {
        'epoch': epoch,
        'best_pre':best_pre,
        'state_dict': cur_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.exists(backupdir):
        os.mkdir(backupdir)
    netname = (netname.split('/')[-1]).split('.')[0]
    # logging('net name: ',netname)
    filepath = backupdir+('/%s_%06d.model'%(netname, epoch))
    torch.save(state, filepath)
    num_classes = cur_model.num_classes
    number_class_target = 3

    if savecondition:
        cur_model.save_weights_tc_as_normal('%s/%s_%06d.weights' % (backupdir, netname, epoch),
                                            number_class_target, num_classes)
    else:
        cur_model.save_weights('%s/%s_%06d.weights' % (backupdir, netname, epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/flir.data', help='data definition file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolov3_flir.cfg', help='network configuration file')
    # parser.add_argument('--weights', '-w', type=str, default=None, help='initial weights file')
    parser.add_argument('--weights', '-w', type=str, default='weights/yolov3.weights', help='initial weights file')
    # parser.add_argument('--continuetrain', '-t', type=str, default='backup/adapter_segment_000017.model', help='load model train')
    parser.add_argument('--continuetrain', '-t', type=str, default=None, help='load model train')
    # parser.add_argument('--eval', '-n', dest='eval', action='store_true', default=True, help='prohibit test evalulation')
    parser.add_argument('--reset', '-r', action="store_true", default=True, help='initialize the epoch and model seen value')
    parser.add_argument('--epoch', '-e', type=int, default=30,help='How many epoch we train, default is 30')
    parser.add_argument('--layerwise', '-l', type=int, default=0, help='Do layerwise for training on number of layer every epoch')
    parser.add_argument('--adaptation', '-a', type=int, default=0,help='Train adaptation freeze some layers')

    FLAGS, _ = parser.parse_known_args()
    main()

#python train.py | grep "avg_loss" log_train.txt