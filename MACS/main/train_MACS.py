import sys
sys.path.append('../utils_labelNoise')
from torch.optim import lr_scheduler
from Encoder.optimizer import MixOptimizer
from Encoder.model import mAtt
from MemoryMoCo import MemoryMoCo
from xbm_memory import *
from utils_noise import *
import torch
import numpy as np
from log_utils import create_logger
import argparse
import logging
import os
import time
from dataset.eeg_dataset import *
from torch import optim
import random
import time

# Assuming `test_loader` is your data loader for the test dataset
# and `model` is your trained model
def test_inference_time_first_batch(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()  # Record start time for the first batch
            output = model(data)  # Model inference
            end_time = time.time()  # Record end time for the first batch
            
            inference_time_batch = end_time - start_time  # Calculate inference time for the batch
            batch_size = data.size(0)  # Get the batch size
            print( batch_size )
            inference_time_per_sample = inference_time_batch / batch_size  # Calculate inference time per sample
            
            print(f"Average Inference Time per Sample for the First Batch: {inference_time_per_sample:.6f} seconds")
            return inference_time_per_sample  # Return after the first batch

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int,default=100)
    parser.add_argument('--cuda_dev', type=int,default=0, help='GPU to select')
    parser.add_argument('--epoch', type=int, default=50, help='training epoches')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--ua_ratio', type=float,
                        default=0.4, help='percent of unreliable annotation')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Beta distribution parameter for switcher')
    parser.add_argument('--network', type=str,
                        default='BNMTrans', help='Network architecture')
    parser.add_argument('--seed_initialization', type=int,
                        default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int,
                        default=42, help='random seed (default: 1)')
    parser.add_argument('--experiment_name', type=str, default='PD',
                        help='name of the experiment (for the output files)')
    parser.add_argument('--dataset', type=str,
                        default='mci', help='mci, pd')
    parser.add_argument('--initial_epoch', type=int, default=1,
                        help="Star training at initial_epoch")
    parser.add_argument('--low_dim', type=int, default=128,
                        help='Size of contrastive learning embedding')
    parser.add_argument('--batch_t', default=0.1, type=float,
                        help='Contrastive learning temperature')
    parser.add_argument('--aprox', type=int, default=1, 
                        help='determine the type of approximation to be applied in the calculation of log_prob')
    parser.add_argument('--memory_use', type=int, default=1, help='1: Use memory')
    parser.add_argument('--memory_begin', type=int, default=1,
                        help='Epoch to begin using memory')
    parser.add_argument('--memory_per_class', type=int, default=100,
                        help='Num of samples per class to store in the memory. Memory size = xbm_per_class*num_classes')
    parser.add_argument('--startLabelCorrection', type=int,
                        default=9999, help='Epoch to start label correction')
    parser.add_argument('--k_val', type=int, default=5,
                        help='k for k-nn correction')
    parser.add_argument('--PredictiveCorrection', type=int,
                        default=0, help='Enable predictive label correction')
    parser.add_argument('--balance_crit', type=str,
                        default="none", help='None, max, min, median')
    parser.add_argument('--discrepancy_corrected', type=int, default=1,
                        help='Use corrected label for discrepancy measure')
    parser.add_argument('--uns_queue_k', type=int, default=300,
                        help='uns-cl num negative sampler')
    parser.add_argument('--uns_t', type=float, default=0.1,
                        help='uns-cl temperature')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='pair selection th')
    parser.add_argument('--m', type=int, default=4, help='temporal scale')
    parser.add_argument('--alpha_moving', type=float,
                        default=0.999, help='exponential moving average weight')
    parser.add_argument('--dim', type=int, default=32, help='encoder dim')
    parser.add_argument('--foldb', type=int, default=1, help='begin fold')

    parser.add_argument('--folde', type=int, default=4, help='endfold')
    parser.add_argument('--channel', type=int, default=63, help='channel number')

    args = parser.parse_args()
    return args


def transform_test(x):
    return x


def data_config(args, transform_train1,transform_train2,transform_test, k):

    trainset, testset, clean_labels, noisy_labels, noisy_indexes, all_labels = get_dataset(
        args, transform_train1, transform_train2,transform_test, k)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels


def main(args):
    for k in range(args.foldb,args.folde):
        best_acc_val = 0.0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed_initialization)  # CPU seed
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

        random.seed(args.seed_initialization)

        # Augmentor
        from augmentations import transform_train1,transform_train2
        train_loader, test_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels = data_config(
            args, transform_train1,transform_train2, transform_test, k)
        st = time.time()

        m = args.m
        model = mAtt(m, num_classes=args.num_classes,
                         low_dim=args.low_dim,dim=args.dim,channel=args.channel).to(device)
        ###moco trick
        model_ema =mAtt(m, num_classes=args.num_classes,
                             low_dim=args.low_dim,dim=args.dim,channel=args.channel).to(device)
        moment_update(model, model_ema, 0)

        print('Total params: {:.2f} M'.format(
            (sum(p.numel() for p in model.parameters()) / 1000000.0)))

        uns_contrast = MemoryMoCo(args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.wd)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer = MixOptimizer(optimizer)

        # Memory creation
        if args.memory_use == 1:
            xbm = Memory(args, device)
        else:
            xbm = []

        loss_train_epoch = []
        loss_val_epoch = []

        acc_train_per_epoch = []
        acc_val_pred_per_epoch = []

        agreement = []
        agreement_measure = []

        exp_path = os.path.join('./out', 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                                    args.seed_initialization,
                                                                                                    args.seed_dataset),
                                str(args.ua_ratio), f'fold{k}')
        res_path = os.path.join('./out', 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                              args.seed_initialization,
                                                                                              args.seed_dataset),
                                str(args.ua_ratio), f'fold{k}')

        if not os.path.isdir(res_path):
            os.makedirs(res_path)

        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)

        logger = create_logger(res_path)
        logger.info(args)

        np.save(res_path + '/' + str(args.ua_ratio) +
                '_true_labels.npy', np.asarray(clean_labels))
        np.save(res_path + '/' + str(args.ua_ratio) +
                '_noisy_labels.npy', np.asarray(noisy_labels))
        np.save(res_path + '/' + str(args.ua_ratio) +
                '_diff_labels.npy', noisy_indexes)
        np.save(res_path + '/' + str(args.ua_ratio) +
                '_all_labels.npy', all_labels)

        cont = 0
        best_acc = 0
        for epoch in range(args.initial_epoch, args.epoch + 1):
            start_time=time.time()
            logger.info(f"=================> {args.experiment_name}   {args.ua_ratio}")
            agreement_measure, soft_selected_pairs, hard_selected_pairs =Stratifier(
                args, model, device, train_loader, test_loader, args.batch_t, epoch,logger)
            scheduler.step()

            loss_per_epoch, train_ac,  train_time = \
                train_MACAC(args, model, model_ema, uns_contrast, device, train_loader, optimizer,
                           epoch, xbm, agreement_measure, soft_selected_pairs, hard_selected_pairs,logger)
            end_time=time.time()
            timeswi=end_time-start_time
            print("11111111111111111111111111111111111",timeswi)

            loss_train_epoch += [loss_per_epoch]

            logger.info('######## Test ########')
            loss_per_epoch_val, acc_val_pred_per_epoch_i, auroc_patient_val, acc_patient_val = test_eval(
                 model, device, test_loader, k,args.dataset,logger)
            avg_inference_time_per_sample = test_inference_time_first_batch(model, device, test_loader)  # Test and log inference time per sample for the first batch
            print("avg_inference_time_per_sample",avg_inference_time_per_sample)
            acc_val_pred_per_epoch_i = [acc_val_pred_per_epoch_i]
            loss_per_epoch_val = [loss_per_epoch_val]

        
            agreement.append(agreement_measure.data.cpu().numpy())

            loss_val_epoch += loss_per_epoch_val
            acc_train_per_epoch += [train_ac]
            acc_val_pred_per_epoch += acc_val_pred_per_epoch_i

            logger.info('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
            st = time.time()

            if acc_patient_val > best_acc or (acc_patient_val == best_acc and auroc_patient_val > best_auroc):
                logger.info(f"Best model saved at epoch {epoch}")
                best_acc, best_auroc = acc_patient_val, auroc_patient_val

            if epoch == args.initial_epoch:
                best_acc_val = acc_val_pred_per_epoch[-1]
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch_val[-1], acc_val_pred_per_epoch[-1], args.ua_ratio, best_acc_val)

                torch.save(model.state_dict(), os.path.join(
                    exp_path, snapBest + '.pth'))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(
                    exp_path, 'opt_' + snapBest + '.pth'))
            else:
                new_val = acc_val_pred_per_epoch[-1]
                if new_val > best_acc_val:
                    best_acc_val = new_val
                    if cont > 0:
                        try:
                            os.remove(os.path.join(
                                exp_path, 'opt_' + snapBest + '.pth'))
                            os.remove(os.path.join(
                                exp_path, snapBest + '.pth'))
                        except OSError:
                            pass

                    snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                        epoch, loss_per_epoch_val[-1], acc_val_pred_per_epoch[-1], args.ua_ratio, best_acc_val)

                    torch.save(model.state_dict(), os.path.join(
                        exp_path, snapBest + '.pth'))
                    torch.save(optimizer.optimizer.state_dict(), os.path.join(
                        exp_path, 'opt_' + snapBest + '.pth'))

            cont += 1
            if (epoch % 5 == 0 or epoch==1):
                snapLast = f"_model{epoch}"
                torch.save(model.state_dict(), os.path.join(
                    exp_path, snapLast + '.pth'))

            if epoch == args.epoch:
                snapLast = f"_model{epoch}"
                torch.save(model.state_dict(), os.path.join(
                    exp_path, snapLast + '.pth'))

            # Save losses:
            np.save(res_path + '/' + str(args.ua_ratio) +
                    '_LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
            np.save(res_path + '/' + str(args.ua_ratio) +
                    '_LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

            # save accuracies:
            np.save(res_path + '/' + str(args.ua_ratio) +
                    '_accuracy_per_epoch_train.npy', np.asarray(acc_train_per_epoch))
          
            np.save(res_path + '/' + str(args.ua_ratio) +
                    '_accuracy_per_epoch_val_pred.npy', np.asarray(acc_val_pred_per_epoch))

          
            np.save(res_path + '/' + 'agreement_per_sample_train.npy',
                    np.asarray(agreement))


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    main(args)
