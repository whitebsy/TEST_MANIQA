import os
import torch
import numpy as np
import logging
import time
import random
import csv
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from data.getting_mos import dis_list3
from model_net.CNN_EPI_SAI import LFIQA
from config import config
from util.options import cv_transform, ToTensor, Normalize

if config.db_name =='win5':
    from data.win5_cnn import IQA_datset
elif config.db_name =='MPI':
    from data.mpi_cnn import IQA_datset
else:
    from data.NBU_cnn import IQA_datset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    idx_epoch = []  # ++

    for data in tqdm(train_loader):

        sai = data['lf_list'][0]
        epi = data['lf_list'][1]
        sai = [x.cuda() for x in sai]
        epi = [x.cuda() for x in epi]
        x_d = [sai, epi]
        # x_d = data['d_img_org'].cuda()
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        idx = data['idx'].cuda()  # ++

        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        idx_batch_numpy = idx.data.cpu().numpy()  # ++
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        idx_epoch = np.append(idx_epoch, idx_batch_numpy)  # ++

    dataPath = config.svPath + '/train/train_pred_{}.csv'.format(epoch + 1)
    with open(dataPath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx_epoch, pred_epoch, labels_epoch))

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        idx_epoch = []  # ++
        # 均值
        count, pred_mean, labels_mean, idx_mean = 0, 0, 0, 0

        for data in tqdm(test_loader):
            pred = 0

            sai = data['lf_list'][0]
            epi = data['lf_list'][1]
            sai = [x.cuda() for x in sai]
            epi = [x.cuda() for x in epi]
            x_d = [sai, epi]
            labels = data['score']
            idx = data['idx'].cuda()  # ++
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred = net(x_d)

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            '''均值'''
            if config.if_avg:
                pred = pred.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                idx = idx.data.cpu().numpy()

                pred_mean += pred
                labels_mean += labels
                idx_mean += idx
                count += 1
                # print(pred_mean, labels_mean, idx_mean)

                if count >= config.avg_num:
                    pred_mean = pred_mean / count
                    labels_mean = labels_mean / count
                    idx_mean = idx_mean / count

                    pred_epoch = np.append(pred_epoch, pred_mean)
                    labels_epoch = np.append(labels_epoch, labels_mean)
                    idx_epoch = np.append(idx_epoch, idx_mean)

                    count, pred_mean, labels_mean, idx_mean = 0, 0, 0, 0

            # save results in one epoch
            else:
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                idx_batch_numpy = idx.data.cpu().numpy()  # ++
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
                idx_epoch = np.append(idx_epoch, idx_batch_numpy)  # ++

        dataPath = config.svPath + '/test/test_pred_{}.csv'.format(epoch + 1)
        with open(dataPath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(idx_epoch, pred_epoch, labels_epoch))

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p))
        print('test epoch:{}  =====  loss:{:.4}  =====  SRCC:{:.4}  =====  PLCC:{:.4}'
              .format(epoch + 1, np.mean(losses), rho_s, rho_p))

        return np.mean(losses), rho_s, rho_p


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    # dis_list
    train_dis, test_dis = dis_list3(config.text_path, train_rate=config.train_rate)

    # data load
    train_transform = torchvision.transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ++
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transforms = torchvision.transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ++
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])

    train_dataset = IQA_datset(
        config=config,
        scene_list=train_dis,
        # transform=train_transform,
        # transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        transform=transforms.Compose([cv_transform()]),
        mode='train',
    )
    val_dataset = IQA_datset(
        config=config,
        scene_list=test_dis,
        # transform=test_transforms,
        # transform=transforms.Compose([Normalize(0.5, 0.5),ToTensor()]),
        transform=transforms.Compose([cv_transform()]),
        mode='test',
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        # batch_size=config.batch_size,
        batch_size=1,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = LFIQA()

    net = net.cuda()

    # loss function
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    for epoch in range(0, config.n_epoch):

        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                # save weights
                # model_name = "epoch{}".format(epoch + 1)
                # model_save_path = os.path.join(config.snap_path, model_name)
                # torch.save(net, model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

