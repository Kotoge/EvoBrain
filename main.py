import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data.dataloader_detection import load_dataset_detection
from data.dataloader_chb import load_dataset_chb
from data.dataloader_prediction import load_dataset_prediction
from args import get_args
from collections import OrderedDict
from json import dumps
from model.gru_gcn import GRU_GCN_classification
from model.DCRNN import DCRNNModel_classification, DCRNNModel_nextTimePred
from model.EvoBrain import EvoBrain_classification
from model.EGCN import EvolveGCN_Model_classification
from model.BIOT import BIOTClassifier
from model.graphs4mer import GraphS4mer
from model.lstm import LSTMModel
from model.cnnlstm import CNN_LSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import pandas as pd
import sklearn
import time
import torch
torch.autograd.set_detect_anomaly(True)

def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = args.device if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, args.dataset, args.task, args.max_seq_len, args.model_name, args.graph_type, args.rand_seed)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    if args.model_name == "BIOT":
        args.use_fft = False

    # Build dataset
    log.info('Building dataset...')
    if args.dataset == 'CHBMIT':
        print("Loading CHBMIT dataset...")
        dataloaders, datasets, scaler = load_dataset_chb(
            task = args.task,
            input_dir=args.input_dir,
            raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            max_seq_len=args.max_seq_len,
            standardize=False,
            num_workers=args.num_workers,
            augmentation=args.data_augment,
            adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            use_fft=args.use_fft,
            sampling_ratio=1,
            seed=123,
            preproc_dir=args.preproc_dir)
    else: #TUSZ
        print("Loading TUSZ dataset...")
        if args.task == 'detection':
            dataloaders, datasets, scaler = load_dataset_detection(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=1,
                seed=123,
                preproc_dir=args.preproc_dir)

        
        elif args.task == 'prediction':
            dataloaders, datasets, scaler = load_dataset_prediction(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=1,
                seed=123,
                preproc_dir=args.preproc_dir)
        else:
            raise NotImplementedError

    # Build model
    log.info('Building model...')
    if args.model_name == "dcrnn":
        model = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "evolvegcn":
        model = EvolveGCN_Model_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "evobrain":
        if args.agg != "max":
            log.info("Using EvoBrain with aggregation method: {}".format(args.agg))
        model = EvoBrain_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "graphs4mer":
        model = GraphS4mer(num_classes=args.num_classes, max_seq_len=args.max_seq_len, num_nodes=args.num_nodes)
    elif args.model_name == "gru_gcn":
        model = GRU_GCN_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "BIOT":
        args.use_fft = False
        model = BIOTClassifier(n_classes=args.num_classes, n_channels=args.num_nodes, n_fft=args.input_dim, hop_length=int(args.input_dim / 2))
    elif args.model_name == "lstm":
        model = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        model = CNN_LSTM(args.num_classes, args.dataset)
    else:
        raise NotImplementedError

    if not args.test:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)
        else:  # fine-tune from pretrained model
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                pretrained_model = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  # placeholder
                pretrained_model = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model)

                model = utils.build_finetune_model(
                    model_new=model,
                    model_pretrained=pretrained_model,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')

        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        model = model.to(device)

        # Train
        train(model, dataloaders, args, device, args.save_dir, log, tbx)

        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    else:
        if args.load_model_path is not None:
            model = utils.load_model_checkpoint(
                args.load_model_path, model)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    model = model.to(device)
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           log,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            log,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    # Define loss function
    if (args.task == 'detection') or (args.task == 'prediction'):
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    memory_usage_list = []
    time_list = []
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for x, y, seq_lengths, supports, adj, file_name in train_loader:
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                supports = supports.to(device)
                adj = adj.to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
                start_time = time.time()
                initial_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

                if args.model_name == "evobrain" or args.model_name == "evolvegcn" or args.model_name == "gru_gcn":
                    logits, _ = model(x, seq_lengths, adj)
                elif args.model_name == "dcrnn":
                    logits, _ = model(x, seq_lengths, supports)     
                elif args.model_name == "BIOT":
                    logits, _ = model(x)  
                elif args.model_name == "lstm" or args.model_name == "cnnlstm" or args.model_name == "graphs4mer":
                    logits, _ = model(x, seq_lengths)
                else:
                    print("model_name: ", args.model_name)
                    raise NotImplementedError
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)          
                loss = loss_fn(logits, y)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()

                end_time = time.time()
                max_memory = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0

                memory_usage_list.append(max_memory - initial_memory)
                time_list.append(end_time - start_time)

                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                if (args.stop == True) and (len(time_list) > 1000):
                    break

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        log,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()

    max_memory_usage = np.max(memory_usage_list) / (1024 ** 2)  
    avg_time_per_batch = np.mean(time_list)

    log.info(f"Average Training Time per Batch: {avg_time_per_batch:.4f} seconds")


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        log,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
    if (args.task == 'detection') or (args.task == 'prediction'):
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    hidden_all = []
    time_list = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, seq_lengths, supports, adj, file_name in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device)  # (batch_size,)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            supports = supports.to(device)
            adj = adj.to(device)

            start_time = time.time()
            # Forward
            # (batch_size, num_classes)
            if args.model_name == "evobrain":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "gru_gcn":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "dcrnn":
                logits, hidden = model(x, seq_lengths, supports)
            elif args.model_name == "evolvegcn":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "BIOT":
                logits, hidden = model(x)
            elif args.model_name == "lstm" or args.model_name == "cnnlstm" or args.model_name == "graphs4mer":
                logits, hidden = model(x, seq_lengths)
            else:
                raise NotImplementedError

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)
            
            time_list.append(time.time() - start_time)
            

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)
            hidden_all.append(hidden.cpu().reshape(hidden.shape[0], -1))

            # Log info
            progress_bar.update(batch_size)
            if (args.stop == True) and (len(time_list) > 1000):
                break

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    hidden_all = np.concatenate(hidden_all, axis=0)
    # print("Hidden shape:", hidden_all.shape)
    # print("y_pred_all shape:", y_pred_all.shape)

    # 評価結果をファイルに保存
    if is_test:
        results_file = os.path.join(save_dir, f'{eval_set}_results.npz')
        np.savez(results_file, 
                 y_true=y_true_all, 
                 y_pred=y_pred_all, 
                 y_prob=y_prob_all, 
                 file_names=file_name_all)
        print(f"Evaluation results saved to {results_file}")

    if eval_set=='test':
            output_file = os.path.join(save_dir, "hidden.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert tensor to DataFrame
            df = pd.DataFrame(hidden_all)
            
            df.to_csv(output_file, mode='w', header=False, index=False)

            output_file = os.path.join(save_dir, "true_labels.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert tensor to DataFrame
            df = pd.DataFrame(np.expand_dims(y_true_all, axis=0))
            
            df.to_csv(output_file, mode='w', header=False, index=False)
    
    avg_time_per_batch = np.mean(time_list)
    log.info(f"Average Test Time per Batch: {avg_time_per_batch:.4f} seconds")


    # Threshold search, for detection only
    if ((args.task == "detection") or (args.task == "prediction")) and (eval_set == 'dev') and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if ((args.task == "detection")or(args.task == "prediction")) else "weighted")

    if args.num_classes == 1 and is_test:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true_all, y_prob_all)
        roc_file = os.path.join(save_dir, f'{eval_set}_roc_data.npz')
        np.savez(roc_file, fpr=fpr, tpr=tpr, thresholds=thresholds)
        print(f"ROC curve data saved to {roc_file}")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results

def check_tensor(data, description):
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{description} is not a tensor! Found type: {type(data)}")


if __name__ == '__main__':
    main(get_args())
