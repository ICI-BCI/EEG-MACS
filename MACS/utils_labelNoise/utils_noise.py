
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import time
from utils.NCECriterion import NCESoftmaxLoss
import pandas as pd
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')



################################# MACAC #############################################
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)
def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)
## Input interpolation functions
def switcher(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam

## Masks creation

## Unsupervised mask for batch and memory (note that memory also contains the current mini-batch)

def unsupervised_masks_estimation(args, xbm, mix_index1, mix_index2, epoch, bsz, device):
    labelsUnsup = torch.arange(bsz).long().unsqueeze(1).to(device)  # If no labels used, label is the index in mini-batch 
    maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).to(device)
    maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
    maskUnsup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    if args.memory_use == 1 and epoch > args.memory_begin:
        ## Extend mask to consider xbm_memory features (all zeros except for the last features stored that contain the augmented view in the memory
        maskUnsup_mem = torch.zeros((2 * bsz, xbm.K)).float().to(device)  ##Mini-batch samples with memory samples (add columns)

        ##Re-use measkUnsup_batch to copy it in the memory (in the righ place) and find the augmented views (without gradients)

        if xbm.ptr == 0:
            maskUnsup_mem[:, -2 * bsz:] = maskUnsup_batch
        else:
            maskUnsup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = maskUnsup_batch

    else:
        maskUnsup_mem = []

    ######################### Mixup additional mask: unsupervised term ######################
    ## With no labels (labelUnsup is just the index in the mini-batch, i.e. different for each sample)
    quad1_unsup = torch.eq(labelsUnsup[mix_index1], labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the dominant)
    quad2_unsup = torch.eq(labelsUnsup[mix_index1], labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    quad3_unsup = torch.eq(labelsUnsup[mix_index2], labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_unsup = torch.eq(labelsUnsup[mix_index2], labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_unsup = torch.cat((quad1_unsup, quad2_unsup), dim=1) #128,256
    mask2_b_unsup = torch.cat((quad3_unsup, quad4_unsup), dim=1)
    mask2Unsup_batch = torch.cat((mask2_a_unsup, mask2_b_unsup), dim=0)

    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Unsup_batch[torch.eye(2 * bsz) == 1] = 0

    if args.memory_use == 1 and epoch > args.memory_begin:
        ## Extend mask to consider xbm_memory features (will be zeros excpet the positions for the augmented views for the second mixup term)
        mask2Unsup_mem = torch.zeros((2 * bsz, xbm.K)).float().to(device)  ##Mini-batch samples with memory samples (add columns)

        if xbm.ptr == 0:
            mask2Unsup_mem[:, -2 * bsz:] = mask2Unsup_batch
        else:
            mask2Unsup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = mask2Unsup_batch

    else:
        mask2Unsup_mem = []


    return maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem


def supervised_masks_estimation(args, labels, index,xbm, xbm_labels, xbm_index,mix_index1, mix_index2, epoch, bsz, device,hard_selected_pairs):
    ###################### Supervised mask  ###############################
    labels = labels.contiguous().view(-1, 1)
    index = index.to(hard_selected_pairs.device)

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')
    temp_graph = hard_selected_pairs[index][:,index]
    ##Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
    # maskSup_batch = torch.eq(labels, labels.t()).float() - torch.eye(bsz, dtype=torch.float32).to(device)
    maskSup_batch = temp_graph.float().to(device) 
    maskSup_batch [torch.eye(bsz) == 1] = 0
    maskSup_batch = maskSup_batch.repeat(2, 2)
    maskSup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    if args.memory_use == 1 and epoch > args.memory_begin:
        ## Extend mask to consider xbm_memory features
        xbm_index = xbm_index.to(hard_selected_pairs.device)
        temp_graph_mem=hard_selected_pairs[index][:,xbm_index]
        # maskSup_mem = torch.eq(labels, xbm_labels.t()).float().repeat(2, 1)  ##Mini-batch samples with memory samples (add columns)
        maskSup_mem = temp_graph_mem.float().repeat(2, 1).to(device)

        if xbm.ptr == 0:
            maskSup_mem[:, -2 * bsz:] = maskSup_batch
        else:
            maskSup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = maskSup_batch

    else:
        maskSup_mem = []

    
    mix_index1 = mix_index1.to(temp_graph.device)
    quad1_sup = temp_graph[mix_index1].float().to(device)
    ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the mayor/dominant)
    quad2_sup = quad1_sup
    ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    mix_index2 = mix_index2.to(temp_graph.device)
    quad3_sup = temp_graph[mix_index2].float().to(device)
    ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_sup = quad3_sup
    ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_sup = torch.cat((quad1_sup, quad2_sup), dim=1)
    mask2_b_sup = torch.cat((quad3_sup, quad4_sup), dim=1)
    mask2Sup_batch = torch.cat((mask2_a_sup, mask2_b_sup), dim=0)


    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Sup_batch[torch.eye(2 * bsz) == 1] = 0

    if args.memory_use == 1 and epoch > args.memory_begin:
      
        maskExtended_sup3_1 = temp_graph_mem[mix_index1].float().to(device)
        ##Mini-batch samples with memory samples (add columns)
        maskExtended_sup3_2 = temp_graph_mem[mix_index2].float().to(device)
        ##Mini-batch samples with memory samples (add columns)
        mask2Sup_mem = torch.cat((maskExtended_sup3_1, maskExtended_sup3_2), dim=0)
        if xbm.ptr == 0:
            mask2Sup_mem[:, -2 * bsz:] = mask2Sup_batch

        else:
            mask2Sup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = mask2Sup_batch

    else:
        mask2Sup_mem = []

    return maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem

#### Losses

def InterpolatedContrastiveLearning_loss(args, pairwise_comp, maskSup, mask2Sup, maskUnsup, mask2Unsup, logits_mask, lam1, lam2, bsz, epoch, device,agreement_measure ):

    logits = torch.div(pairwise_comp, args.batch_t)

    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-10)

    # compute mean of log-likelihood over positive (weight individual loss terms with mixing coefficients)

    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    ## Second mixup term log-probs
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))
    mean_log_prob_pos2_unsup = (mask2Unsup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))

    ## Weight first and second mixup term (both data views) with the corresponding mixing weight

    ##First mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss1a = -lam1 * mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] - lam1 * mean_log_prob_pos_sup[:int(len(mean_log_prob_pos_sup) / 2)]
    ##First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss1b =-lam2 * mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] - lam2 * mean_log_prob_pos_sup[int(len(mean_log_prob_pos_sup) / 2):]
    ## All losses for first mixup term
    loss1 = torch.cat((loss1a, loss1b))

    ##Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss2a = -(1.0 - lam1)* mean_log_prob_pos2_unsup[:int(len(mean_log_prob_pos2_unsup) / 2)] - (1.0 - lam1) * mean_log_prob_pos2_sup[:int(len(mean_log_prob_pos2_sup) / 2)]
    ##Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss2b =-(1.0 - lam2)* mean_log_prob_pos2_unsup[int(len(mean_log_prob_pos2_unsup) / 2):] - (1.0 - lam2) * mean_log_prob_pos2_sup[int(len(mean_log_prob_pos2_sup) / 2):]
    ## All losses secondfor first mixup term
    loss2 = torch.cat((loss2a, loss2b))

    ## Final loss (summation of mixup terms after weighting)
    loss = loss1 + loss2

    loss = loss.view(2, bsz).mean(dim=0)
    loss = ((maskSup[:bsz].sum(1))>0)*(loss.view(bsz))
    return loss.mean()

## Semi-supervised learning

def ClassificationLoss(args, predsA, predsB, predsNoDA, y_a1, y_b1, y_a2, y_b2, mix_index1, mix_index2, lam1, lam2, criterionCE, agreement_measure, epoch, device):

    preds = torch.cat((predsA, predsB), dim=0)

    targets_1 = torch.cat((y_a1, y_a2), dim=0)
    targets_2 = torch.cat((y_b1, y_b2), dim=0)
    mix_index = torch.cat((mix_index1, mix_index2), dim=0)

    ones_vec = torch.ones((predsA.size()[0],)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device) #256

    if args.PredictiveCorrection == 0 or epoch <= args.startLabelCorrection:
        loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)
        loss = loss.mean()


    elif args.PredictiveCorrection == 1 and epoch > args.startLabelCorrection:
        agreement_measure = torch.cat((agreement_measure, agreement_measure), dim=0)
        lossLabeled = agreement_measure * (
                    lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2))
        lossLabeled = lossLabeled.mean()

        ## Pseudo-labeling
        prob = F.softmax(predsNoDA, dim=1)
        prob = torch.cat((prob, prob), dim=0)
        z1 = prob.clone().detach()
        z2 = z1[mix_index, :]
        preds_logSoft = F.log_softmax(preds)

        loss_x1_pred_vec = lam_vec * (1 - agreement_measure) * (-torch.sum(z1 * preds_logSoft, dim=1))  ##Soft
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_pred_vec = (1 - lam_vec) * (1 - agreement_measure[mix_index]) * (
            -torch.sum(z2 * preds_logSoft, dim=1))  ##Soft
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        lossUnlabeled = loss_x1_pred + loss_x2_pred


        loss = lossLabeled + lossUnlabeled

    return loss


def train_MACAC(args, model, model_ema,uns_contrast,device, train_loader, optimizer, epoch, xbm, agreement,soft_selected_pairs ,hard_selected_pairs,logger):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    acc = AverageMeter()
    # switch to train mode
    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1
    criterion = NCESoftmaxLoss(reduction="none").cuda()

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")

    for batch_idx, (eeg1, eeg2, eeg_noDA, labels, _, index, _, clean_labels) in enumerate(train_loader):
        agreement_measure = agreement[index]
        eeg1, eeg2, eeg_noDA, labels, index = eeg1.to(device), eeg2.to(device), eeg_noDA.to(device), labels.to(device), index.to(device)

        model.zero_grad()

        ##uns-cl
        _,feat_q = model(eeg1)
       
        with torch.no_grad():
            _, feat_k= model_ema(eeg2)
        out = uns_contrast(feat_q, feat_k, feat_k, update=True)
        uns_loss = criterion(out).mean()  

        ##Interpolated inputs
        eeg1, y_a1, y_b1, mix_index1, lam1 = switcher(eeg1, labels, args.alpha, device)
        eeg2, y_a2, y_b2, mix_index2, lam2 = switcher(eeg2, labels, args.alpha, device)
        bsz = eeg1.shape[0]
        start_time=time.time()
        predsA, embedA = model(eeg1)
        predsB, embedB = model(eeg2)

        ## Forward pass free of DA
        predsNoDA, _ = model(eeg_noDA)


        ## Compute classification loss (returned individual per-sample loss)

        ## Remove this preds from graph
        predsNoDA = predsNoDA.detach()
        start_time = time.time() 
    
        ############# Update memory ##############
        if args.memory_use == 1:
            xbm.enqueue_dequeue(torch.cat((embedA.detach(), embedB.detach()), dim=0), torch.cat((labels.detach().squeeze(), labels.detach().squeeze()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0))

        ############# Get features from memory ##############
        if args.memory_use == 1 and epoch > args.memory_begin:
            xbm_feats, xbm_labels,xbm_index = xbm.get()
            xbm_labels = xbm_labels.unsqueeze(1)
        else:
            xbm_feats, xbm_labels ,xbm_index=  torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        #####################################################

        ###################### Unsupervised mask with augmented view ###############################
        maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, xbm, mix_index1, mix_index2, epoch, bsz, device)
        ############################################################################################

        ## Contrastive learning
        embeds_batch = torch.cat([embedA, embedB], dim=0)
        pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())#256,256

        if args.memory_use == 1 and epoch > args.memory_begin:
            embeds_mem = torch.cat([embedA, embedB, xbm_feats], dim=0)
            pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz], embeds_mem[2 * bsz:].t()) ##Compare mini-batch with memory
            ######################################################################

        ###################### Supervised mask  ###############################
        maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem = \
            supervised_masks_estimation(args, labels, index.long() ,xbm, xbm_labels, xbm_index.long(),mix_index1, mix_index2, epoch, bsz, device,hard_selected_pairs)
        
        ############################################################################################

        lossClassif = ClassificationLoss(args, predsA, predsB, predsNoDA, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                        mix_index2, lam1, lam2, criterionCE, agreement_measure, epoch, device)
        

        # Mask-out self-contrast cases
        logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(device))  ## Negatives mask, i.e. all except self-contrast sample

        loss_sup = InterpolatedContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz, epoch, device,agreement_measure )

        if args.memory_use == 1 and epoch > args.memory_begin:

            logits_mask_mem = torch.ones_like(maskSup_mem) ## Negatives mask, i.e. all except self-contrast sample

            if xbm.ptr == 0:
                logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
            else:
                logits_mask_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = logits_mask_batch

            loss_mem = InterpolatedContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz, epoch, device,agreement_measure)

            loss_sup = loss_sup + loss_mem
            sel_mask=(maskSup_batch[:bsz].sum(1)+maskSup_mem[:bsz].sum(1))<2

        
        else:
        
            sel_mask=(maskSup_batch[:bsz].sum(1))<1

        
        sel_loss = (sel_mask*uns_loss).mean() + loss_sup
        
        loss = sel_loss + lossClassif

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        moment_update(model, model_ema, args.alpha_moving)

        prec1, prec5 = accuracy_v2(predsNoDA, labels, top=[1, 2])
        acc.update(prec1.item(), eeg1.size(0))
        train_loss.update(loss.item(), eeg1.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 5 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(eeg1), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                optimizer.optimizer.param_groups[0]['lr']))
        counter = counter + 1

    return train_loss.avg, acc.avg,  batch_time.sum


def collect_metrics(dataset,y_probs_test, y_true_test, y_pred_test, sample_indices_test,
					fold_idx,mode):

	dataset_index = pd.read_csv(f"corpus_window_index_{dataset}.csv", dtype={"patient_id":str, })

	# create patient-level train and test dataframes
	rows = [ ]
	for i in range(len(sample_indices_test)):
		idx = sample_indices_test[i]
		# print(idx)
		temp = { }
		temp["patient_id"] = str(dataset_index.loc[idx, "patient_id"])
		temp["sample_idx"] = idx
		temp["y_true"] = y_true_test[i]
		temp["y_probs_0"] = y_probs_test[i, 0]
		temp["y_probs_1"] = y_probs_test[i, 1]
		temp["y_pred"] = y_pred_test[i]
		rows.append(temp)
	test_patient_df = pd.DataFrame(rows)

	# get patient-level metrics from window-level dataframes
	y_probs_test_patient, y_true_test_patient, y_pred_test_patient = get_patient_prediction(test_patient_df, fold_idx)


	# PATIENT-LEVEL ROC PLOT - select optimal threshold for this, and get patient-level precision, recall, f1
	fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,1])

	# select an optimal threshold using the ROC curve
	# Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
	optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]

	# calculate class predictions and confusion-based metrics using the optimal threshold
	roc_predictions = [1 if i >=optimal_proba_cutoff else 0 for i in y_probs_test_patient[:,1]]
	precision_patient_test =  precision_score(y_true_test_patient, roc_predictions)
	recall_patient_test =  recall_score(y_true_test_patient, roc_predictions)
	f1_patient_test = f1_score(y_true_test_patient, roc_predictions)
	acc_patient_test = accuracy_score(y_true_test_patient, roc_predictions)
       

	auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,1])

	auroc_test = roc_auc_score(y_true_test, y_probs_test[:,1])
	 
	precision_test = precision_score(y_true_test, y_pred_test)
	recall_test = recall_score(y_true_test, y_pred_test)
	f1_test = f1_score(y_true_test, y_pred_test)
	acc_test = accuracy_score(y_true_test, y_pred_test)
	
	return auroc_test,precision_test, recall_test, f1_test,acc_test,auroc_patient_test,precision_patient_test, recall_patient_test, f1_patient_test, acc_patient_test

# create subject-level metrics
def get_patient_prediction(df, fold_idx):
	unique_patients = list(df["patient_id"].unique())
	grouped_df = df.groupby("patient_id")
	rows = [ ]
	for patient in unique_patients:
		patient_df = grouped_df.get_group(patient)
		temp = { }
		temp["patient_id"] = patient
		temp["y_true"] = list(patient_df["y_true"].unique())[0]
		assert len(list(patient_df["y_true"].unique())) == 1
		temp["y_pred"] = patient_df["y_pred"].mode()[0]
		temp["y_probs_0"] = patient_df["y_probs_0"].mean()
		temp["y_probs_1"] = patient_df["y_probs_1"].mean()
		rows.append(temp)
	return_df = pd.DataFrame(rows)
	return np.array(list(zip(return_df["y_probs_0"], return_df["y_probs_1"]))), list(return_df["y_true"]), list(return_df["y_pred"])

def test_eval(model, device, test_loader,k,dataset,logger):
    model.eval()
    loss_per_batch = []

    test_loss = 0
    total=0
    correct=0
    y_probs_val = torch.empty(0, 2).to(device)
    y_true_val = [ ]
    y_pred_val = [ ]
    indices= np.load(f"{dataset}/testindices{k}.npy",allow_pickle=True)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data)
            
            outputs = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(outputs, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(outputs, target).item())
       
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            y_true_val += target.cpu().numpy().tolist()
            y_probs_val = torch.cat((y_probs_val, output.data), 0)
            y_pred_val += predicted.cpu().numpy().tolist()
          
    y_probs_val = torch.nn.functional.softmax(y_probs_val, dim=1).cpu().numpy()
    y_true_val = np.array(y_true_val)

    auroc_val,precision_val, recall_val, f1_val, acc_val,auroc_patient_val,  precision_patient_val, recall_patient_val, f1_patient_val, acc_patient_val= collect_metrics(dataset,y_probs_test=y_probs_val,
                        y_true_test=y_true_val,
                        y_pred_test=y_pred_val,
                        sample_indices_test =indices,					
                        fold_idx=k,
                        mode='test')

   
    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set prediction branch: Average loss: {:.4f},  Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.info(f"Validation metrics: AUROC{auroc_val:.4f}, precision: {precision_val:.4f}, recall: {recall_val:.4f}, f1: {f1_val:.4f}, acc: {acc_val:.4f}")
    logger.info(f"Validation patient metrics: AUROC{auroc_patient_val:.4f}, precision: {precision_patient_val:.4f}, recall: {recall_patient_val:.4f}, f1: {f1_patient_val:.4f}, acc: {acc_patient_val:.4f}")
    loss_per_epoch = np.average(loss_per_batch)
    # print(test_loss,loss_per_epoch)
    acc_val_per_epoch = np.array(100. * correct / len(test_loader.dataset))
    return loss_per_epoch,acc_val_per_epoch,auroc_patient_val,acc_patient_val

def Stratifier(args, net, device, trainloader, testloader, sigma, epoch,logger):

    net.eval()

    cls_time = AverageMeter()
    end = time.time()

    
    trainLabels = torch.LongTensor(trainloader.dataset.targets).to(device)

    C = trainLabels.max() + 1

    ## Get train features
    transform_bak1 = trainloader.dataset.transform1
    transform_bak2 = trainloader.dataset.transform2

    trainloader.dataset.transform1 = testloader.dataset.transform
    trainloader.dataset.transform2 = testloader.dataset.transform

    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=8)

    trainFeatures = torch.rand(len(trainloader.dataset), args.low_dim).t().to(device)
    smiliar_graph_all=torch.zeros(len(trainloader.dataset),len(trainloader.dataset))#3786，3786
    
    for batch_idx, (inputs, _, _, noisyLabels, _, index, _, targets) in enumerate(temploader):
        inputs = inputs.to(device)
        batchSize = inputs.size(0)

        _, features = net(inputs)
        trainFeatures[:, batch_idx * 100:batch_idx * 100 + batchSize] = features.data.t()

    trainLabels = torch.LongTensor(temploader.dataset.clean_labels).to(device)
    trainNoisyLabels = torch.LongTensor(temploader.dataset.targets).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.targets).to(device)


    discrepancy_measure1 = torch.zeros((len(temploader.dataset.targets),)).to(device)
    discrepancy_measure2 = torch.zeros((len(temploader.dataset.targets),)).to(device)
    discrepancy_measure1_pseudo_labels = torch.zeros((len(temploader.dataset.targets),)).to(device)#3786
    discrepancy_measure2_pseudo_labels = torch.zeros((len(temploader.dataset.targets),)).to(device)

    agreement_measure = torch.zeros((len(temploader.dataset.targets),))#.to(device)

    ## Weighted k-nn correction

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs, _, _, targets, _, index, _, _) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            smiliar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size()[0]), index] = -1 ##Self-contrast set to -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1) ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi) ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
            # (set of neighbouring labels) is turned into a one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_() ## Apply temperature to scores
            yd_transform[...] = 1.0 ##To avoid using similarities
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]#100,2

            prob_temp = probs_norm[torch.arange(0, batchSize), targets]#100
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[index] = -torch.log(prob_temp)

            if args.discrepancy_corrected == 0:
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1]==targets).float().data.cpu()


            _, predictions_corrected = probs_corrected.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels

            cls_time.update(time.time() - end)


    tran_new_labels2 = train_new_labels.clone()
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs, _, _, targets, _, index, _, clean_labels) in enumerate(temploader):

            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = tran_new_labels2.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
            # (set of neighbouring labels) is turned into a one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities only counts
            probs_corrected = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)

            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]


            prob_temp = probs_norm[torch.arange(0, batchSize), targets]
            prob_temp[prob_temp<=1e-2] = 1e-2
            prob_temp[prob_temp > (1-1e-2)] = 1-1e-2

            discrepancy_measure2[index] = -torch.log(prob_temp)

            if args.discrepancy_corrected == 1:
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1]==targets).float().data.cpu()


            cls_time.update(time.time() - end)


    #### Set balanced criterion for noise detection
    if args.balance_crit == "max" or args.balance_crit =="min" or args.balance_crit =="median":
        #agreement_measure_balanced = torch.zeros((len(temploader.dataset.targets),)).to(device)
        num_clean_per_class = torch.zeros(args.num_classes)
        for i in range(args.num_classes):
            idx_class = temploader.dataset.targets==i
            idx_class = torch.from_numpy(idx_class.astype("float")) == 1.0
            num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])

        if args.balance_crit =="max":
            num_samples2select_class = torch.max(num_clean_per_class)
        elif args.balance_crit =="min":
            num_samples2select_class = torch.min(num_clean_per_class)
        elif args.balance_crit =="median":
            num_samples2select_class = torch.median(num_clean_per_class)

        agreement_measure = torch.zeros((len(temploader.dataset.targets),)).to(device)

        for i in range(args.num_classes):
            idx_class = temploader.dataset.targets==i
            samplesPerClass = idx_class.sum()
            idx_class = torch.from_numpy(idx_class.astype("float"))# == 1.0
            idx_class = (idx_class==1.0).nonzero().squeeze()
            if args.discrepancy_corrected == 0:
                discrepancy_class = discrepancy_measure1[idx_class]
            else:
                discrepancy_class = discrepancy_measure2[idx_class]

            if num_samples2select_class>=samplesPerClass:
                k_corrected = samplesPerClass
            else:
                k_corrected = num_samples2select_class

            top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1]
            idx_class = idx_class.to(agreement_measure.device)
            ##Agreement measure sets to 1 those samples detected as clean
            agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0

    selected_examples=agreement_measure
    logger.info(f'selected examples {sum(selected_examples)}')
    trainloader.dataset.transform1 = transform_bak1
    trainloader.dataset.transform2 = transform_bak2

    if sum(selected_examples) == 0:
        return False
    ## select pairs 
    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples,as_tuple=True)[0].cpu()
        total_selected_num=len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)
        total_num = len(trainNoisyLabels)
        
       
        noisy_pairs=torch.eq(trainNoisyLabels, trainNoisyLabels.t()) 
        hard_select_pairs=torch.zeros_like(noisy_pairs, dtype=torch.bool)
        selected_pairs = noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)].clone() #1990 1990
        hard_select_pairs[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)] = selected_pairs
        temp_graph = smiliar_graph_all[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)]         
        selected_th=np.quantile(temp_graph[selected_pairs],args.beta)
        logger.info(f'selected_th {selected_th}')
        temp = torch.zeros(total_num,total_num).type(torch.uint8)
       
        
        noisy_pairs = torch.where(smiliar_graph_all<selected_th,temp,noisy_pairs.type(torch.uint8)).type(torch.bool)
        noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)] = selected_pairs
        final_selected_pairs = noisy_pairs     
        #### through experiement ,we do not use final_selected_pairs 
        
    return selected_examples.cuda(),final_selected_pairs.contiguous(),hard_select_pairs.contiguous()


