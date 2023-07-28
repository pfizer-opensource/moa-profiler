"""
Script containing core definitions and classes for the classification task
"""
import pandas as pd
import pickle
import torch
import os
import numpy as np
import math
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import time
import shutil
import sklearn.preprocessing
import sklearn.decomposition 
import sklearn.manifold
import matplotlib
import matplotlib.pyplot as plt
import re
import random
import sqlite3
import gzip
import csv as csv_w ##careful about defining this globally since "csv" is a common variable name 
import itertools
import scipy.stats
import multiprocessing as mp
import heapq
from torchvision import datasets, transforms
import sklearn.metrics
import matplotlib.colors as colors
from datetime import date, datetime
from collections import Counter
import torch.distributed as dist
import socket
import torch.multiprocessing as tmp
from contextlib import closing
from collections import OrderedDict
import argparse
from efficientnet_pytorch import * 
from sklearn import svm
import copy
import scipy.spatial
from sklearn.linear_model import LogisticRegression
import psutil

def getRowColumn(imagename):
    """
    Given image name, will return the row and column
    as r##c##
    """
    matches = re.findall(r'r[0-9][0-9]c[0-9][0-9]', imagename)
    assert(len(matches) == 1) ##should only match once else ambiguous
    return matches[0]

def getField(imagename):
    matches =  re.findall(r'f[0-9][0-9]', imagename)   
    return matches[0]

def getBarcode(imagename):
    """
    Given image name, will return the barcode as BR######## (JUMP data) or SQ######## (lincs data)
    """
    r = re.compile(r'BR[0-9]+ | SQ[0-9]+', flags=re.I | re.X)
    matches =r.findall(imagename)
    if len(matches) == 0:
        raise Exception("no barcode found for {}".format(imagename))
    assert(len(set(matches)) == 1) ##should only match once else ambiguous
    return matches[0] 

def getJumpBatch(imagename):
    batches = ["2020_11_04_CPJUMP1", "2020_12_02_CPJUMP1_2WeeksTimePoint", 
    "2020_11_18_CPJUMP1_TimepointDay1", "2020_12_07_CPJUMP1_4WeeksTimePoint","2020_11_19_TimepointDay4", "2020_12_08_CPJUMP1_Bleaching"]
    for batch in batches:
        if batch in imagename:
            return batch
    return "no batch"

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
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

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on train mode
    model.train()
    end = time.time()
    start_time = time.time()
    counter = 0 
    for batch_idx, (_, input, target, _) in enumerate(loader):
        # Create vaiables
        input = input.cuda()
        target = target.cuda()
        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)
        if torch.isnan(loss):
            print("nan loss!")
            continue
        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    # Return summary statistics
    print("training time for this epoch:{}".format(time.time() - start_time))
    return batch_time.avg, losses.avg, error.avg

def test_epoch(model, loader, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    k_accuracies = AverageMeter() ##to keep track of top-5 accuracy 
    # Model on eval mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (_, input, target, _) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            if torch.isnan(loss):
                print("validation nan loss!")
                continue
            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1) ##output = batch x # classes, pred = indices of max element for each instance
            
            ##top-5 accuracy
            if is_test:
                _, pred_k = output.data.cpu().topk(k=5, dim=1) ##pred = indices of top k elements for each instance
                k_accuracy = sum([1 for i in range(0, len(target.cpu())) if target.cpu()[i] in pred_k[i]])  / batch_size
                k_accuracies.update(k_accuracy, batch_size)
            
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size) ##error is 1 - accuracy
            losses.update(loss.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                print(res)
                if is_test:
                    print("k accuracy: ", k_accuracy)
            
        if is_test:
            print("Final Avg Test Error: ", round(error.avg, 3), "Avg Accuracy: ", round(1 - error.avg, 3))
            print("top-5 accuracy: {}".format(k_accuracies.avg))
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg

def getClassificationStats(model, loader, label_index_map=None):
    """
    Prints confusion matrix and class-specific reports
    Return class specific accuracy map 
    """
    reverse_map = {value: key for (key, value) in label_index_map.items()}
    model.eval()
    scores, predictions, labels = [], [], []
    with torch.no_grad():
        for batch_idx, (_, input, target, _) in enumerate(loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1) ##output = batch x # classes, pred = indices of max element for each instance
            batch_preds = pred.squeeze().tolist()
            batch_preds = [reverse_map[p] for p in batch_preds]
            batch_labels = target.tolist()
            batch_labels = [reverse_map[l] for l in batch_labels]
            scores += output.tolist()
            predictions += batch_preds
            labels += batch_labels
    classification_map = {"labels": labels, "predictions": predictions, "scores":scores, "label_index_map":label_index_map}
    return classification_map 

def makeSaveDir(save, tagline):
    """
    Make directory SAVE and README.txt file with TAGLINE string
    """
    if not os.path.exists(save):
        os.makedirs(save)
        os.mkdir(save + "models/")
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    with open(os.path.join(save, 'README.txt'), 'w') as f:
        f.write(tagline + '\n')
    print("made save_dir: ", save)
    
def train(model=None, train_loader=None, valid_loader=None, test_loader=None, save=None, n_epochs=300, batch_size=16, lr=0.1, wd=0.0001, momentum=0.9, seed=None, print_freq=10, tagline=None):
    """
    method for training model, assumes that we are NOT using DistributedDataParallel
    """
    if seed is not None:
        torch.manual_seed(seed)
    makeSaveDir(save, tagline)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')
    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            print_freq=print_freq
        )
        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=valid_loader,
            is_test=(not valid_loader),
            print_freq=print_freq
        )
        # Determine if model is the best
        if valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'models/model_best.dat'))
        torch.save(model.state_dict(), os.path.join(save, 'models/model_{}.dat'.format(epoch + 1)))
        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'models/model_best.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True,
        print_freq=print_freq
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)

class JUMPMOADataset(torch.utils.data.Dataset):
    """
    For JUMP data with MOA annotations
    """
    def __init__(self, csv_file, transform, jitter=False, label_index_map=None, augment=False, reverse=False):
        df = pd.read_csv(csv_file)
        self.data = df.values
        self.transform = transform
        self.label_index_map = label_index_map
        self.augment = augment
        self.reverse = reverse
        self.jitter = jitter
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imagename, broad_sample, perturbation_t, cell_type, gene_targets, control_type, perturbation, moas = self.data[idx]
        ch1 = Image.open(imagename)
        ch2 = Image.open(imagename.replace("ch1", "ch2"))
        ch3 = Image.open(imagename.replace("ch1", "ch3"))
        ch4 = Image.open(imagename.replace("ch1", "ch4"))
        ch5 = Image.open(imagename.replace("ch1", "ch5"))
        if self.jitter: ##brightness and contrast jitters
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3) 
            ch1 = jitter(ch1)
            ch2 = jitter(ch2)
            ch3 = jitter(ch3)
            ch4 = jitter(ch4)
            ch5 = jitter(ch5)
        dest = np.zeros((5, 1080, 1080), dtype=np.float32)
        dest[0,:] = (ch1 - np.amin(ch1)) / (np.amax(ch1) + .000001)
        dest[1,:] = (ch2 - np.amin(ch2)) / (np.amax(ch2) + .000001)
        dest[2,:] = (ch3 - np.amin(ch3)) / (np.amax(ch3) + .000001)
        dest[3,:] = (ch4 - np.amin(ch4)) / (np.amax(ch4) + .000001)
        dest[4,:] = (ch5 - np.amin(ch5)) / (np.amax(ch5) + + .000001)
        dest = torch.from_numpy(dest)
        if self.transform != None:
            dest = self.transform(dest) 
        ##reverse channel order if reverse is True
        if self.reverse:
            dest = torch.flip(dest, [0])
        if self.augment:
            random_num = random.randint(0,3)
            dest = torch.rot90(dest, k=random_num, dims=[1,2]) ##random 90 degree rotation 
            if random_num >= 2:##horizontal flip at random
                dest = torch.fliplr(dest)
        return imagename, dest, self.label_index_map[moas], perturbation

class LINCSClassificationDataset(torch.utils.data.Dataset):
    """
    For single label LINCS data
    """
    def __init__(self, csv_file, transform, jitter=False, label_index_map=None, augment=False, reverse=False):
        df = pd.read_csv(csv_file)
        self.data = df.values
        self.transform = transform
        self.label_index_map = label_index_map
        self.augment = augment
        self.reverse = reverse
        self.jitter = jitter
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imagename,gene_targets,moas,concentration,perturbation = self.data[idx]
        ch1 = Image.open(imagename)
        ch2 = Image.open(imagename.replace("ch1", "ch2"))
        ch3 = Image.open(imagename.replace("ch1", "ch3"))
        ch4 = Image.open(imagename.replace("ch1", "ch4"))
        ch5 = Image.open(imagename.replace("ch1", "ch5"))
        if self.jitter: ##brightness and contrast jitters
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3) 
            ch1 = jitter(ch1)
            ch2 = jitter(ch2)
            ch3 = jitter(ch3)
            ch4 = jitter(ch4)
            ch5 = jitter(ch5)
        dest = np.zeros((5, 1080, 1080), dtype=np.float32)
        dest[0,:] = (ch1 - np.amin(ch1)) / (np.amax(ch1) + .000001)
        dest[1,:] = (ch2 - np.amin(ch2)) / (np.amax(ch2) + .000001)
        dest[2,:] = (ch3 - np.amin(ch3)) / (np.amax(ch3) + .000001)
        dest[3,:] = (ch4 - np.amin(ch4)) / (np.amax(ch4) + .000001)
        dest[4,:] = (ch5 - np.amin(ch5)) / (np.amax(ch5) + + .000001)
        dest = torch.from_numpy(dest)
        if self.transform != None:
            dest = self.transform(dest)
        ##reverse channel order if reverse is True
        if self.reverse:
            dest = torch.flip(dest, [0])
        if self.augment:
            random_num = random.randint(0,3)
            dest = torch.rot90(dest, k=random_num, dims=[1,2]) ##random 90 degree rotation 
            if random_num >= 2:##horizontal flip at random
                dest = torch.fliplr(dest) 
        return imagename, dest, self.label_index_map[moas], perturbation

class FourChannelLINCSClassificationDataset(torch.utils.data.Dataset):
    """
    For four channel LINCS data that excludes "ch3" of the dataset (i.e. RNA)
    """
    def __init__(self, csv_file, transform, jitter=False, label_index_map=None, augment=False, reverse=False):
        df = pd.read_csv(csv_file)
        self.data = df.values
        self.transform = transform
        self.label_index_map = label_index_map
        self.augment = augment
        self.reverse = reverse
        self.jitter = jitter
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imagename,gene_targets,moas,concentration,perturbation = self.data[idx]
        ch1 = Image.open(imagename)
        ch2 = Image.open(imagename.replace("ch1", "ch2"))
        ch4 = Image.open(imagename.replace("ch1", "ch4"))
        ch5 = Image.open(imagename.replace("ch1", "ch5"))
        if self.jitter: ##brightness and contrast jitters
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3) 
            ch1 = jitter(ch1)
            ch2 = jitter(ch2)
            ch4 = jitter(ch4)
            ch5 = jitter(ch5)
        dest = np.zeros((4, 1080, 1080), dtype=np.float32)
        dest[0,:] = (ch1 - np.amin(ch1)) / (np.amax(ch1) + .000001)
        dest[1,:] = (ch2 - np.amin(ch2)) / (np.amax(ch2) + .000001)
        dest[2,:] = (ch4 - np.amin(ch4)) / (np.amax(ch4) + .000001)
        dest[3,:] = (ch5 - np.amin(ch5)) / (np.amax(ch5) + .000001)
        dest = torch.from_numpy(dest)
        if self.transform != None:
            dest = self.transform(dest)
        ##reverse channel order if reverse is True
        if self.reverse:
            dest = torch.flip(dest, [0])
        if self.augment:
            random_num = random.randint(0,3)
            dest = torch.rot90(dest, k=random_num, dims=[1,2]) ##random 90 degree rotation 
            if random_num >= 2:##horizontal flip at random
                dest = torch.fliplr(dest) 
        return imagename, dest, self.label_index_map[moas], perturbation

class FourChannelJUMPClassificationDataset(torch.utils.data.Dataset):
    """
    For four channel JUMP data that excludes "ch3" of the dataset (i.e. RNA)
    """
    def __init__(self, csv_file, transform, jitter=False, label_index_map=None, augment=False, reverse=False):
        df = pd.read_csv(csv_file)
        self.data = df.values
        self.transform = transform
        self.label_index_map = label_index_map
        self.augment = augment
        self.reverse = reverse
        self.jitter = jitter
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imagename, broad_sample, perturbation_t, cell_type, gene_targets, control_type, perturbation, moas = self.data[idx]
        ch1 = Image.open(imagename)
        ch2 = Image.open(imagename.replace("ch1", "ch2"))
        ch4 = Image.open(imagename.replace("ch1", "ch4"))
        ch5 = Image.open(imagename.replace("ch1", "ch5"))
        if self.jitter: ##brightness and contrast jitters
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3) 
            ch1 = jitter(ch1)
            ch2 = jitter(ch2)
            ch4 = jitter(ch4)
            ch5 = jitter(ch5)
        dest = np.zeros((4, 1080, 1080), dtype=np.float32)
        dest[0,:] = (ch1 - np.amin(ch1)) / (np.amax(ch1) + .000001)
        dest[1,:] = (ch2 - np.amin(ch2)) / (np.amax(ch2) + .000001)
        dest[2,:] = (ch4 - np.amin(ch4)) / (np.amax(ch4) + .000001)
        dest[3,:] = (ch5 - np.amin(ch5)) / (np.amax(ch5) + .000001)
        dest = torch.from_numpy(dest)
        if self.transform != None:
            dest = self.transform(dest)
        ##reverse channel order if reverse is True
        if self.reverse:
            dest = torch.flip(dest, [0])
        if self.augment:
            random_num = random.randint(0,3)
            dest = torch.rot90(dest, k=random_num, dims=[1,2]) ##random 90 degree rotation 
            if random_num >= 2:##horizontal flip at random
                dest = torch.fliplr(dest) 
        return imagename, dest, self.label_index_map[moas], perturbation

class CombinedDataset(torch.utils.data.Dataset):
    """
    For combined JUMP1 and lincs dataset
    """
    def __init__(self, csv_file, transform, label_index_map=None, augment=False):
        df = pd.read_csv(csv_file)
        self.data = df.values
        self.transform = transform
        self.label_index_map = label_index_map
        self.augment = augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imagename, dataset, perturbation, concentration, moas = self.data[idx]
        if dataset == "lincs":
            ch1 = Image.open(imagename)
            ch2 = Image.open(imagename.replace("ch1", "ch2"))
            ch3 = Image.open(imagename.replace("ch1", "ch3"))
            ch4 = Image.open(imagename.replace("ch1", "ch4"))
            ch5 = Image.open(imagename.replace("ch1", "ch5"))
        if dataset == "JUMP1": ##let's stick with lincs channel ordering, so reverse JUMP1 ordering
            ch1 = Image.open(imagename)
            ch2 = Image.open(imagename.replace("ch1", "ch4"))
            ch3 = Image.open(imagename.replace("ch1", "ch3"))
            ch4 = Image.open(imagename.replace("ch1", "ch2"))
            ch5 = Image.open(imagename.replace("ch1", "ch1"))
        dest = np.zeros((5, 1080, 1080), dtype=np.float32)
        dest[0,:] = (ch1 - np.amin(ch1)) / (np.amax(ch1) + .000001)
        dest[1,:] = (ch2 - np.amin(ch2)) / (np.amax(ch2) + .000001)
        dest[2,:] = (ch3 - np.amin(ch3)) / (np.amax(ch3) + .000001)
        dest[3,:] = (ch4 - np.amin(ch4)) / (np.amax(ch4) + .000001)
        dest[4,:] = (ch5 - np.amin(ch5)) / (np.amax(ch5) + + .000001)
        dest = torch.from_numpy(dest)
        dest = self.transform(dest) 
        if self.augment:
            random_num = random.randint(0,3)
            dest = torch.rot90(dest, k=random_num, dims=[1,2]) ##random 90 degree rotation 
            if random_num >= 2:##horizontal flip at random
                dest = torch.fliplr(dest) 
        return imagename, dest, self.label_index_map[moas], perturbation

def getLatentRepresentations(study=None, well_aggregator=None, model=None, loader=None, cardinality=None, label_index_map=None):
    """
    Will save a dictionary of latent state representations as a dictionary
    Will save two dictionaries, one with each image as its own embedding, and one that is well-aggregated (one embedding per well)
    CARDINALITY is the dimension of the latent representation
    LABEL_TYPE is used for book keeping and determining the name to save the pkl file
    """
    model.eval()
    imagenames_list = []
    targets = torch.Tensor()
    embeddings = torch.empty(0, cardinality)
    perturbations_list = []
    i = 0
    length = len(loader)
    with torch.no_grad():
        for batch_idx, (imagenames, input, target, perturbations) in enumerate(loader):
            if batch_idx == len(loader) - 2:
                print('RAM % used:', psutil.virtual_memory()[2], psutil.virtual_memory()[3]/1000000000)
            input = input.cuda()
            latent_rep = model.getLatentRepresentation(input).cpu() ##offload from GPU memory
            imagenames_list += imagenames
            perturbations_list += perturbations
            targets = torch.cat([targets, target])
            embeddings = torch.cat([embeddings, latent_rep])
            i += 1
    targets = targets.detach().numpy()
    embeddings = embeddings.detach().numpy()
    labels = [] ##string label representations of targets
    for i in range(0, len(targets)):
        label = [item[0] for item in label_index_map.items() if item[1] == targets[i]]
        assert(len(label) == 1) ##should only be one match
        labels.append(label[0])
    assert(len(targets) == len(embeddings) == len(labels))
    latent_dictionary = {"targets":targets, "embeddings":embeddings, "labels":labels, "imagenames":imagenames_list, "perturbations":perturbations_list}
    
    if study == "lincs":
        barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb"))

    ##write a raw un-aggregated version
    # unaggregated_latent_dictionary = latent_dictionary

    ##aggregate by well and write a new dictionary
    return_targets = []
    return_labels = []
    return_embeddings = []
    return_wells = [] ##keep track of which well an embedding belongs to for book-keeping
    return_perturbations = []
    well_dictionary = {} ##need to keep track of which embeddings belong to which well, key: well (barcode + row_col), value: list of [np arrays of tensor embeddings, target(singleton)]
    for i in range(0, len(imagenames_list)):
        if study == "JUMP1":
            well = getJumpBatch(imagenames_list[i]) + getBarcode(imagenames_list[i]) + getRowColumn(imagenames_list[i])
        if study == "lincs":
            well = barcode_to_platemap[getBarcode(imagenames_list[i])]  + getBarcode(imagenames_list[i]) + getRowColumn(imagenames_list[i]) 
        if well not in well_dictionary:
            well_dictionary[well] = [[embeddings[i]], targets[i], labels[i], perturbations_list[i]] ##init as [list, target, label, perturbation]
        else:
            well_dictionary[well][0].append(embeddings[i])
            assert(targets[i] == well_dictionary[well][1]) ##make sure that target is the same - should be for fields in the same well
            assert(labels[i] == well_dictionary[well][2]) ##make sure that string label is the same - should be for fields in the same well
            assert(perturbations_list[i] == well_dictionary[well][3]) 
    for well in well_dictionary: ##finally aggregate the entries and append to return tensors to get well-aggregated embeddings
        return_wells.append(well)
        if well_aggregator == "mean":
            return_embeddings.append(np.mean(well_dictionary[well][0],axis=0))
        if well_aggregator == "median":
            return_embeddings.append(np.median(well_dictionary[well][0],axis=0))
        if well_aggregator == "pca":
            return_embeddings.append(PCAAggregate(well_dictionary[well][0]))
        return_targets.append(well_dictionary[well][1])
        return_labels.append(well_dictionary[well][2])
        return_perturbations.append(well_dictionary[well][3])
    targets = np.array(return_targets)
    embeddings = np.array(return_embeddings)
    assert(len(targets) == len(embeddings))
    latent_dictionary = {"targets":targets, "embeddings":embeddings, "labels":return_labels, "wells": return_wells, "perturbations":return_perturbations}
    return latent_dictionary

def extractProfilerRepresentations(study="JUMP1", method=None, loader=None, deep_profile_type=None):
    """
    label_type is used for book-keeping and determining file name to save
    METHOD either "cellProfiler" or "deepProfiler"
    Will run through LOADER and based on imagename, will pull CellProfiler representation of the well that captures the image from the map pickles/{study}/cellProfilerFeatures.pkl
    Saves a dictionary called "pickles/{study}/CP_latent_dictionary_label_type_{}_{}.pkl" with keys: "targets" and "embeddings"
    """
    ##load metadata mapping dictionaries
    if study == "JUMP1":
        batch_well_to_sample = pickle.load(open("pickles/JUMP1/batch_well_to_sample.pkl", "rb"))
        sample_to_gene_target = pickle.load(open("pickles/JUMP1/sample_to_gene_target.pkl", "rb"))
        sample_to_compound = pickle.load(open("pickles/JUMP1/broad_sample_to_compound_map.pkl", "rb"))
        compound_to_moa = pickle.load(open("pickles/JUMP1/compound_to_moa_map.pkl", "rb"))
    if study == "lincs":
        barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb"))
        platemap_well_to_sample_and_concentration = pickle.load(open("pickles/lincs/lincs_platemap_well_to_sample_and_concentration.pkl", "rb"))
        lincs_sample_to_moa_target_pert = pickle.load(open("pickles/lincs/lincs_sample_to_moa_target_pert.pkl", "rb"))
    row_index_to_letter = pickle.load(open("pickles/JUMP1/row_index_to_letter.pkl", "rb"))
    ##load profile maps
    if method == "cellProfiler":
        if study == "JUMP1":
            profile_map = pickle.load(open("pickles/{}/cellProfilerFeatures.pkl".format(study), "rb"))
        if study == "lincs":
            profile_map = pickle.load(open("pickles/{}/cellProfilerFeatures_from_repo_level_3.pkl".format(study), "rb"))
    if method == "deepProfiler":
        profile_map = pickle.load(open("pickles/{}/deepProfilerFeatures_from_{}.pkl".format(study, deep_profile_type), "rb"))
    ##iterate over loader 
    embeddings = []
    labels = []
    return_wells = []
    return_perturbations = []
    embedded = {} ##key: plate + "|" + barcode + "|" + well, value: vector embedding 
    counter, counter_two = 0, 0
    missed_wells = set()
    with torch.no_grad():
        for batch_idx, (imagenames, input, target, perturbations) in enumerate(loader):
            for j in range(0, len(imagenames)):
                counter += 1
                barcode = getBarcode(imagenames[j])
                well = getRowColumn(imagenames[j]) ##format: r##c##
                letter_well = row_index_to_letter[int(well[1:3])] + well[4:]
                ##get label of this image
                if study == "JUMP1":
                    batch = getJumpBatch(imagenames[j])
                    sample = batch_well_to_sample[batch, "compound", letter_well] 
                    label = compound_to_moa[sample_to_compound[sample]]
                    assert(len(label) == 1)
                    label = list(label)[0]
                if study == "lincs":
                    platemap = barcode_to_platemap[barcode]
                    sample = platemap_well_to_sample_and_concentration[platemap + "_" + letter_well][0]
                    if sample == "NoSample":
                        label = "Empty"
                    else:
                        assert(len(lincs_sample_to_moa_target_pert[sample]["moa"]) == 1) ##assert only 1 moa present
                        label = lincs_sample_to_moa_target_pert[sample]["moa"][0]
                if study == "JUMP1":
                    key = batch + "|" + barcode + "|" + well
                if study == "lincs":
                    key = platemap + "|" + barcode + "|" + well
                if key in profile_map:
                    vector = profile_map[key]
                else:##key not present, either control, or missing data
                    if label != "Empty" or label != "no_target":
                        # print("non-control missing key for LINCS cellProfiler: ", key)
                        counter_two += 1
                        missed_wells.add(key)
                    continue
                ##don't want to add embeddings for wells we already added, check to make sure they are the same 
                if key in embedded:
                    assert(np.array_equal(embedded[key], vector))
                else:
                    embeddings.append(vector) ##technically this vector corresponds to the whole well, not just this image, therefore only take one embedding per well
                    labels.append(label)
                    return_wells.append(key.replace("|", ""))
                    return_perturbations.append(perturbations[j])
                    embedded[key] = vector
    embeddings = np.array(embeddings)
    assert(len(embeddings) == len(labels))
    print(study, method, ": number of distinct targets: ", len(set(labels)), ", # images: ", counter, ", misses: ", counter_two, ", missed wells: ", len(missed_wells), ", extracted wells: ", len(set(return_wells)), ", length of loader: ", len(loader), "number of embeddings: ", len(embeddings))
    latent_dictionary = {"embeddings": embeddings, "labels":labels, "wells": return_wells, "perturbations": return_perturbations}
    return latent_dictionary

def euclideanDist(tup):
    """
    Helper function for parallel compute to return euclidean distance of tup[0] tup[1]
    """
    return np.linalg.norm(tup[0] - tup[1])

def pearson(tup):
    """
    Helper function for parallel compute to return pearson between tup[0] tup[1]
    """
    return -1 * scipy.stats.pearsonr(tup[0], tup[1])[0] ## we'll multiply by -1 because we are using a heap that prioritizes lower value -> best pearson will be close to -1 now 

def dropNegative(embeddings, labels):
    """
    Will remove "Empty" and "no_target" embeddings,
    returns new embeddings and labels
    """
    purge_indices = []
    for i in range(0, len(labels)):
        if labels[i] in ["no_target", "Empty"]: ##JUMP represent with "no_target", lincs uses "Empty"
            purge_indices.append(i)
    embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
    labels = [labels[j] for j in range(0, len(labels)) if j not in purge_indices]
    return embeddings, labels

def getNeighborAccuracy(latent_dictionary, metric="pearson", verbose=False, drop_neg_control=False):
    """
    Get percentage of population who are accurately classified by assigning the majority label of the k nearest neighbors
    Also prints replicate and non-replicate correlation
    metric = "distance" for euclidean distance
    or "pearson" for pearson correlation coefficient 
    if VERBOSE will print information
    if DROP_NEG_CONTROL will remove entries where target == no_target (or rather index of "no_target" in gene map)
    label_type will determine which pickle file to load, and name to save
    Returns a map with key: k, value: accuracy
    """    
    embeddings = copy.deepcopy(latent_dictionary["embeddings"])
    labels = copy.deepcopy(latent_dictionary["labels"])
    print("original size: ", len(labels))
    if drop_neg_control:
        embeddings, labels = dropNegative(embeddings, labels)
        print("dropped negative controls, new size: ", len(labels))
    assert(embeddings.shape[0] == len(labels))
    cardinality = len(labels)
    class_sizes = {l: labels.count(l) for l in labels} ##key: label, value: count of label in labels 
    ##key: index of labels list, value: heap (list) of (metric score, class, other index)
    ##define "dissimilar" = more positive distance = more positive pearson (so we'll need take the negative pearson)
    closest_neighbor_map = {i:[] for i in range(0, len(labels))}
    start_time = time.time()
    #evaluate distances and add to heaps for each index
    #need to do full iteration, can't simply have unique (i,j) sets
    for i in range(0, len(labels)):
        for j in range(0, len(labels)):
            if i != j:
                if metric == "distance":
                    score = np.linalg.norm(embeddings[i] - embeddings[j])
                if metric == "pearson":
                    score = -1 * scipy.stats.pearsonr(embeddings[i], embeddings[j])[0] ## we'll multiply by -1 because we are using a heap that prioritizes lower value -> best pearson will be close to -1 now 
                if metric == "cosine":
                    score = -1 * (1.0 - scipy.spatial.distance.cosine(embeddings[i], embeddings[j]))
                heapq.heappush(closest_neighbor_map[i],(score, labels[j], j))
    
    ##now iterate over indices and get: kNN accuracy, replicate correlation, non-replicate correlation  
    nonreplicate_correlations = [] 
    replicate_correlations = {label: [] for label in set(labels)} ##key: label, value: list of pairwise pearson correlations of embeddings with label=label
    ks = range(1, cardinality - 1)
    accuracy_map = {k:[0, 0] for k in ks} ##key: k, value: [total matches, total mismatches]
    k_pred_labels_map = {k: {"predictions": [], "labels": []} for k in ks} ##key k, key: "predictions" or "labels", value: list of multi-class strings
    k_stats = {k: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for k in ks} ##map key k, key: stat metric, value: count, for the inverse k-nn problem "I predict that my k closest neighbors share my class"
    counter = 0
    for index in closest_neighbor_map:
        counter += 1
        my_class = labels[index]
        my_heap = closest_neighbor_map[index]
        popped_heap = [heapq.heappop(my_heap) for i in range(0, len(my_heap))] 
        ##fill in kNN accuracy map and stats
        for k in ks:
            k_nearest = popped_heap[0:k]
            classes = [entry[1] for entry in k_nearest]
            majority_k = Counter(classes).most_common()[0][0] ##if tied count, then Counter will prioritize by value first encountered, which will be the 0th element anyway because of heap ordering - which is what we want!
            if my_class != majority_k:
                accuracy_map[k][1] += 1
            else:
                accuracy_map[k][0] += 1
            k_pred_labels_map[k]["predictions"].append(majority_k)
            k_pred_labels_map[k]["labels"].append(my_class)
        ##fill in replicate dictionary and non-replicate list
        for i in range(0, len(popped_heap)):
            entry_score, entry_class, entry_index = popped_heap[i]
            if entry_class == my_class:
                replicate_correlations[entry_class].append(-1 * entry_score) ##-1 undoes original negation
            else:
                nonreplicate_correlations.append(-1 * entry_score)
        ##fill in k_stats: TP, FP, TN, FN 
        matches = [1 if popped_heap[i][1] == my_class else 0 for i in range(0, len(popped_heap))] ##list of 1s and 0s such that a 1 at index i indicates a class-match at popped_heap[i] 
        for k in ks:
        # for k in range(1, class_sizes[my_class]): ##don't want to use full range of k because for large ks, we quickly build up large FPs even if perfectly clustered...go up to my class size
            TPs = sum(matches[0:k]) ##all the 1s within a k neighborhood range
            FPs = k - TPs
            FNs = sum(matches[k:]) ##all the 1s outside of k 
            TNs = len(matches) - k - FNs
            k_stats[k]["TP"] += TPs
            k_stats[k]["FP"] += FPs 
            k_stats[k]["FN"] += FNs
            k_stats[k]["TN"] += TNs
            assert(TPs + FPs + FNs + TNs == len(matches))
    ##condense and write accuracy maps
    for k in accuracy_map:
        accuracy_map[k] = round(accuracy_map[k][0] / float((accuracy_map[k][0] + accuracy_map[k][1])), 3)       
    ##average replicate and non-replicate correlations
    nonreplicate_correlation_mean, nonreplicate_correlation_std = np.mean(nonreplicate_correlations), np.std(nonreplicate_correlations)
    replicate_correlations = {key: replicate_correlations[key] for key in replicate_correlations if len(replicate_correlations[key]) != 0} ##some classes only have one total embedding (n=1), therefore some lists will be empty and we should delete from replicate_correlations dict (no replicates)
    for key in replicate_correlations:
        replicate_correlations[key] = np.mean(replicate_correlations[key])
    replicate_correlation_mean, replicate_correlation_std = np.mean([value for value in replicate_correlations.values()]), np.std([value for value in replicate_correlations.values()])
    replicate_correlation_map = {"replicate": (replicate_correlation_mean, replicate_correlation_std), "nonreplicate": (nonreplicate_correlation_mean, nonreplicate_correlation_std)}
    if verbose:
        print("getNeighborAccuracy results: ")
        print("    time elapsed: {} \n".format(round(time.time() - start_time, 2)))
        sorted_accuracy_map =  sorted(accuracy_map.items(), key= lambda x:x[1])
        print("    sorted accuracy map: ", sorted_accuracy_map[0:10], "...", sorted_accuracy_map[-10:])
        print("    non-replicate correlation avg: ", round(nonreplicate_correlation_mean, 3), round(nonreplicate_correlation_std, 3))
        print("    average replicate pearson over all labels: ", round(replicate_correlation_mean, 2), round(replicate_correlation_std, 2))
    return accuracy_map, k_pred_labels_map, k_stats, replicate_correlation_map

def getAccuracyByClosestAggregateLatent(latent_dictionary, class_aggregator=None, metric=None, drop_neg_control=None):
    """
    Stipulates the prediction class of an embedding as the class with the most similar average latent representation
    Writes dictionaries with key: "predictions" or "labels", value: list of multi-class strings
    """
    embeddings = copy.deepcopy(latent_dictionary["embeddings"])
    labels = copy.deepcopy(latent_dictionary["labels"])
    if drop_neg_control:
        embeddings, labels = dropNegative(embeddings, labels)
        print("dropped negative controls, new size: ", len(labels))
    assert(embeddings.shape[0] == len(labels))
    print("getAccuracyByClosestAggregateLatent results:")
    print("    total embeddings: ", embeddings.shape[0])
    ##instantiate class label to average embedding map 
    class_to_count = {c: labels.count(c) for c in set(labels)}
    assert(sum(class_to_count.values())==len(labels))
    class_to_aggregate_embedding = {} 
    class_to_unaggregated_embeddings = {} ##used when aggregator == median or PCA, need to store unaggregated to compute medians without self 
    if class_aggregator == "mean":
        for i in range(0, len(labels)):
            if labels[i] not in class_to_aggregate_embedding:
                class_to_aggregate_embedding[labels[i]] = embeddings[i]
            else:
                class_to_aggregate_embedding[labels[i]] = (embeddings[i] + class_to_aggregate_embedding[labels[i]]) / 2. ##average 
    if class_aggregator in ["median", "pca"]:
        for i in range(0, len(labels)):
            if labels[i] not in class_to_aggregate_embedding:
                class_to_aggregate_embedding[labels[i]] = [embeddings[i]]
                class_to_unaggregated_embeddings[labels[i]] = [embeddings[i]]
            else:
                class_to_aggregate_embedding[labels[i]].append(embeddings[i])
                class_to_unaggregated_embeddings[labels[i]].append(embeddings[i])
        if class_aggregator == "median":
            class_to_aggregate_embedding = {c: np.median(class_to_aggregate_embedding[c], axis=0) for c in class_to_aggregate_embedding}
        if class_aggregator == "pca":
            class_to_aggregate_embedding = {c: PCAAggregate(class_to_aggregate_embedding[c]) for c in class_to_aggregate_embedding}

    ##iterate over original embeddings and get the class prediction based on closeness to aggregated embeddings 
    pred_labels_map = {"predictions": [], "labels": []} ##key: "predictions" or "labels", value: list of multi-class strings
    for i in range(0, len(embeddings)):
        if class_to_count[labels[i]] <= 1:
            print("skipping embedding with label={}, n=1".format(labels[i]))
            continue
        highest_score = 0
        pred_class = ""
        ##evaluate candidate aggregated embeddings and find best pred
        for class_label in class_to_aggregate_embedding:
            if labels[i] == class_label: ##if my class is the same as the candidate class, adjust mean or median to NOT include self so long as there is more than 1 embedding
                if class_aggregator == "mean":
                    aggregate_embedding = ((class_to_aggregate_embedding[class_label] * class_to_count[class_label]) - embeddings[i]) / float(class_to_count[class_label] - 1)   
                if class_aggregator in ["median", "pca"]:
                    unaggregated_embeddings = class_to_unaggregated_embeddings[class_label] 
                    unaggregated_embeddings = [x for x in unaggregated_embeddings if not np.array_equal(x, embeddings[i])]
                    if class_aggregator == "median":
                        aggregate_embedding = np.median(unaggregated_embeddings, axis=0)
                    if class_aggregator == "pca":
                        aggregate_embedding = PCAAggregate(unaggregated_embeddings)
            else:
                aggregate_embedding = class_to_aggregate_embedding[class_label]     
            if metric == "pearson":
                score = scipy.stats.pearsonr(embeddings[i], aggregate_embedding)[0]
            if metric == "distance":
                score = -1 * np.linalg.norm(embeddings[i] - aggregate_embedding) ##minimize positive distance = maximize negative distance
            if metric == "cosine":
                score = 1.0 - scipy.spatial.distance.cosine(embeddings[i], aggregate_embedding)
            if score >= highest_score:
                highest_score = score
                pred_class = class_label
        pred_labels_map["labels"].append(labels[i])
        pred_labels_map["predictions"].append(pred_class)
    return pred_labels_map 

def getEnrichment(latent_dictionary, metric="pearson",  drop_neg_control=None):
    """
    Saves enrichment score of latent embeddings:
    Get pairwise similarity of all embeddings (i, j, pearson)
    take top 1% of most similar, call this highly correlated, everything else weakly correlated
    make 2 x 2 table:
                       MOA replica | non-MOA replica
    highly correlated         a             b
    weakly correlated         c             d
    enrichment = (a/b) / (c/d)
    """
    embeddings = copy.deepcopy(latent_dictionary["embeddings"])
    labels = copy.deepcopy(latent_dictionary["labels"])
    if drop_neg_control:
        embeddings, labels = dropNegative(embeddings, labels)
        print("dropped negative controls, new size: ", len(labels))
    assert(embeddings.shape[0] == len(labels))
    index_combos = itertools.combinations(range(0, len(labels)), 2)
    if metric == "pearson":
        entries = [(i,j, scipy.stats.pearsonr(embeddings[i], embeddings[j])[0]) for (i,j) in index_combos] ##list of (i,j,pearson) tuples
    if metric == "cosine":
        entries = [(i,j, 1.0 - scipy.spatial.distance.cosine(embeddings[i], embeddings[j])) for (i,j) in index_combos] ##list of (i,j,pearson) tuples
    all_similarities = [entry[2] for entry in entries]
    sorted_similarities = np.array(sorted(all_similarities))
    percentiles = list(np.arange(98, 100, .2))
    enrichment_dict = {perc: "" for perc in percentiles}
    for perc in percentiles:
        top_threshold = np.percentile(sorted_similarities, perc) ##percentile value
        highly_correlated = [entry for entry in entries if entry[2] >= top_threshold]
        weakly_correlated = [entry for entry in entries if entry[2] < top_threshold]
        a = sum([1 for entry in highly_correlated if labels[entry[0]] == labels[entry[1]] ])
        b = sum([1 for entry in highly_correlated if labels[entry[0]] != labels[entry[1]] ])
        c = sum([1 for entry in weakly_correlated if labels[entry[0]] == labels[entry[1]] ])
        d = sum([1 for entry in weakly_correlated if labels[entry[0]] != labels[entry[1]] ])
        enrichment = float(a/b) / float(c/d)
        enrichment_dict[perc] = enrichment
    return enrichment_dict

def findFreePort():
    """
    Finds an available port and returns it as a string
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup(rank, world_size, master_addr, master_port):
    """
    sets the os environment master address and master port to MASTER_ADDR and MASTER_PORT
    also initializes a process group with backend gloo, init method env://, RANK, and WORLD_SIZE
    backend nccl was not working with 4 GPUs and stalled (max was 3/4 GPUs), gloo used as of 5/24/22
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(
    backend='gloo',
    init_method='env://',
    world_size=world_size,
    rank=rank
    )
    print("initialized process group gloo")

def train_distributed(gpu, model=None, train_set=None, valid_loader=None, test_loader=None, save=None, n_epochs=300, batch_size=16, lr=0.1, wd=0.0001, momentum=0.9, seed=None, print_freq=10, tagline=None, world_size=None, nr=None, gpus=None, master_addr=None, master_port=None, train_num_workers=None):
    """
    Training function for PyTorch specific DistributedDataParallel
    """
    rank = nr * gpus + gpu
    setup(rank, world_size, master_addr, master_port) 
    torch.cuda.set_device(gpu)
    model=model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    if seed is not None:
        torch.manual_seed(seed)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)
    ##distributed data parallel specific samplers and loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        shuffle=True,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True, num_workers=train_num_workers, sampler=train_sampler)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')
    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            print_freq=print_freq
        )
        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=valid_loader,
            is_test=(not valid_loader),
            print_freq=print_freq
        )
        # Determine if model is the best
        if valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'models/model_best.dat'))
        torch.save(model.state_dict(), os.path.join(save, 'models/model_{}.dat'.format(epoch + 1)))
        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
    # Final test of model on test set
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True,
        print_freq=print_freq
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)
    
def getStratifiedPerformance(model=None, loader=None, stratify_by="well", label_index_map=None):
    """
    Get performance by some stratification criteria
    Stratify by either "plate", "plate_MOA", "well", "perturbation", (if study==lincs) "concentration", (if study==JUMP1) "cell_type", "timepoint"
    Saves dictionary 
    """
    cell_type_time_map = pickle.load(open("pickles/JUMP1/barcode_to_cell_type_and_time.pkl", "rb")) ##only used for JUMP stratification
    reverse_map = {value: key for (key, value) in label_index_map.items()}
    if stratify_by == "concentration":
        img_name_to_conc = {}
        df = pd.read_csv("csvs/lincs/lincs_no_polypharm.csv") ## "JUMP1 study only has 1 concentration! Can't stratify by concentration"
        for index, row in df.iterrows():
            img_name_to_conc[row["imagename"]] = row["concentration"]

    stratified_performance_dict = {} ##if stratify_by == "perturbation": key:perturbation, value: prediction accuracy over all image fields for that perturbation, length
                                     ##if stratify_by == "well": key: well position, value: prediction accuracy over all image fields at that well, length
                                     ##if stratify_by == "concentration": key: concentration, value: prediction accuracy over all image fields at that concentration, length
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (imagenames, input, target, perturbation) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1) ##output = batch x # classes, pred = indices of max element for each instance
            for j in range(0, len(target)):
                if stratify_by == "plate_MOA":
                    key = getBarcode(imagenames[j]) + "_" + reverse_map[target[j].item()] ##plate_MOA 
                if stratify_by == "plate":
                    key = getBarcode(imagenames[j])
                if stratify_by == "well":
                    key = getRowColumn(imagenames[j]) ##well
                if stratify_by == "perturbation":
                    key = perturbation[j]
                if stratify_by == "concentration":
                    key = img_name_to_conc[imagenames[j]]
                if stratify_by == "cell_type":
                    cell_type, timepoint = cell_type_time_map[getJumpBatch(imagenames[j]) + "|" + getBarcode(imagenames[j])]
                    key = cell_type
                if stratify_by == "timepoint":
                    cell_type, timepoint = cell_type_time_map[getJumpBatch(imagenames[j]) + "|" + getBarcode(imagenames[j])]
                    key = timepoint

                if key not in stratified_performance_dict:
                    if target[j] == pred.squeeze()[j]: ##if prediction is correct
                        stratified_performance_dict[key]=[1]
                    else:
                        stratified_performance_dict[key]=[0]
                else:
                    if target[j] == pred.squeeze()[j]:
                        stratified_performance_dict[key].append(1)
                    else:
                        stratified_performance_dict[key].append(0)
    for key in stratified_performance_dict:
        stratified_performance_dict[key] = (sum(stratified_performance_dict[key]) / float(len(stratified_performance_dict[key])), len(stratified_performance_dict[key]))
    return stratified_performance_dict

def getNormalizedPixelDistributions(loader=None, study=None):
    """
    Calculates and plots a normalized histogram of pixel values returned from the LOADER  
    """
    channel_map = {i: [] for i in range(0, 5)}
    with torch.no_grad():
        for batch_idx, (imagenames, input, target, perturbation) in enumerate(loader):
            input = input.cuda()
            for i in range(0, len(input)):
                for j in range(0, 5): 
                    channel_map[j].append(input.cpu().detach().numpy()[i][j])
    channel_map = {key: np.array(channel_map[key]).flatten() for key in channel_map}
    stats_map = {i: {"mean": "", "std": "", "min":"", "max":""} for i in range(0,5)}
    for channel in channel_map:
        stats_map[channel]["mean"] =  round(np.mean(channel_map[channel]), 2)
        stats_map[channel]["std"] =  round(np.std(channel_map[channel]), 2)
        stats_map[channel]["min"] =  round(np.amin(channel_map[channel]), 2)
        stats_map[channel]["max"] =  round(np.amax(channel_map[channel]), 2)
    fig, ax = plt.subplots()
    color_map = {0:"red", 1:"purple",2:"gold", 3:"blue", 4:"grey"}
    height = .98
    for channel in channel_map:
        plt.hist(channel_map[channel], alpha=0.5, label=channel, density=True, color=color_map[channel])
        for stat in stats_map[channel]:
            ax.annotate("{} {}: {}".format(channel, stat, stats_map[channel][stat]), xy=(0, height),fontsize=5)
            height -= .04
    ax.set_ylabel("Frequency")
    ax.set_ylim((0, 1.03))
    ax.set_xlabel("Normalized Pixel Value")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":10}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.title("{}: Histogram of Normalized Pixel\nValues By Channel".format(study))
    plt.savefig("outputs/normalized_pixel_values_{}.png".format(study), dpi=300)

def removeEmptyLabels(latent_dictionary):
    """
    Creates a deep copy of latent_dictionary, removes any entries where latent_dictionary["labels"] == "Empty" or "no_target"
    Returns copy 
    """
    ##exclude label=Empty or label=unknown entries from latent_dictionary 
    latent_copy = copy.deepcopy(latent_dictionary)
    indices_to_keep = []
    for i in range(0, len(latent_copy["labels"])):
        if latent_copy["labels"][i] not in ["Empty", "no_target"]:
            indices_to_keep.append(i)
    for key in latent_copy:
        latent_copy[key] = [latent_copy[key][i] for i in indices_to_keep]
    return latent_copy

def PCAAggregate(X):
    if isinstance(X, list):
        X_copy = np.array(X)
    else:
        X_copy = X
    standardized = (X_copy - X_copy.mean(axis=0)) / (X_copy.std(axis=0) + .0000000000001) ##add small offset in case std is 0
    pca = sklearn.decomposition.PCA()
    X_transformed = pca.fit_transform(standardized)
    return pca.components_[0] ##return first eigenvector

def compoundHoldoutClassLatentAssignment(latent_dictionary, study=None, class_aggregator=None, metric=None, drop_neg_control=None, training_compounds=None, label_index_map=None):
    if drop_neg_control:
        latent_dictionary = removeEmptyLabels(latent_dictionary)
    ##partition latent_dictionary into training and holdout 
    training_latent = {"labels": [], "embeddings":[], "perturbations":[]}
    holdout_latent = {"labels": [], "embeddings":[], "perturbations":[]}
    for i in range(0, len(latent_dictionary["perturbations"])):
        if latent_dictionary["perturbations"][i] in training_compounds:
            training_latent["labels"].append(latent_dictionary["labels"][i])
            training_latent["perturbations"].append(latent_dictionary["perturbations"][i])
            training_latent["embeddings"].append(latent_dictionary["embeddings"][i])
        else:
            holdout_latent["labels"].append(latent_dictionary["labels"][i])
            holdout_latent["perturbations"].append(latent_dictionary["perturbations"][i])
            holdout_latent["embeddings"].append(latent_dictionary["embeddings"][i])
    ##aggregate training embeddings to MOA level instead of well level 
    class_to_aggregate_embedding = {} 
    training_labels = training_latent["labels"]
    training_embeddings = training_latent["embeddings"]
    for i in range(0, len(training_labels)):
        if training_labels[i] not in class_to_aggregate_embedding:
            class_to_aggregate_embedding[training_labels[i]] = [training_embeddings[i]]
        else:
            class_to_aggregate_embedding[training_labels[i]].append(training_embeddings[i])
    if class_aggregator == "median":
        class_to_aggregate_embedding = {c: np.median(np.array(class_to_aggregate_embedding[c]), axis=0) for c in class_to_aggregate_embedding}
    if class_aggregator == "pca":
        class_to_aggregate_embedding = {c: PCAAggregate(np.array(class_to_aggregate_embedding[c])) for c in class_to_aggregate_embedding}
    holdout_embeddings = holdout_latent["embeddings"]
    holdout_labels = holdout_latent["labels"]
    holdout_perturbations = holdout_latent["perturbations"]

    ##iterate over holdout embeddings and get the class prediction based on closeness to training embeddings 
    ##pred_labels_map will be at the well-level
    pred_labels_map = {"perturbations": [], "predictions": [], "labels": []} ##key: "predictions" or "labels", value: list of multi-class strings
    for i in range(0, len(holdout_embeddings)):
        highest_score = 0
        pred_class = ""
        ##evaluate candidate aggregated embeddings and find best pred
        for class_label in class_to_aggregate_embedding:
            aggregate_embedding = class_to_aggregate_embedding[class_label]     
            if metric == "pearson":
                score = scipy.stats.pearsonr(holdout_embeddings[i], aggregate_embedding)[0]
            if metric == "cosine":
                score = 1.0 - scipy.spatial.distance.cosine(holdout_embeddings[i], aggregate_embedding)
            if metric == "distance":
                score = -1 * np.linalg.norm(holdout_embeddings[i] - aggregate_embedding) ##minimize positive distance = maximize negative distance
            if score >= highest_score:
                highest_score = score
                pred_class = class_label
        pred_labels_map["labels"].append(holdout_labels[i])
        pred_labels_map["predictions"].append(pred_class)
        pred_labels_map["perturbations"].append(holdout_perturbations[i])

    ##vote by well and see performance 
    ##since pred_labels_map is at the well-level, let's take votes by well for each 
    perturbation_to_label = {} ##key: perturbation, value: MOA label 
    perturbation_to_well_predictions = {} ##key: perturbation, value: list of MOA well-predictions
    for i in range(0, len(pred_labels_map["labels"])):
        pert, label, pred = pred_labels_map["perturbations"][i], pred_labels_map["labels"][i], pred_labels_map["predictions"][i]
        if pert not in perturbation_to_label:
            perturbation_to_label[pert] = label
        else:
            assert(perturbation_to_label[pert] == label)
        if pert not in perturbation_to_well_predictions:
            perturbation_to_well_predictions[pert] = [pred]
        else:
            perturbation_to_well_predictions[pert].append(pred)
    ##reassign perturbation_to_well_predictions values to Counter object
    perturbation_to_well_predictions = {key: Counter(perturbation_to_well_predictions[key]).most_common() for key in perturbation_to_well_predictions}
    ##do top-k
    reverse_map = {value: key for (key, value) in label_index_map.items()}
    classes = []
    for i in range(0, len(reverse_map)):
        classes.append(reverse_map[i])
    k_map = {} ##key: k, value: ((f1, precision,recall, accuracy), (prediction, labels))
    largest_k = 24 if study == "JUMP1" else 11 ##JUMP1 compounds were mostly at 23 compound replicates, LINCS had a range from 0 -> 10 
    for k in range(1, largest_k):
        predictions, labels = [], []
        for pert in perturbation_to_well_predictions:
            true_label = perturbation_to_label[pert]
            top_k = getTopKIncludingTies(perturbation_to_well_predictions[pert], k)
            if true_label in top_k:
                predictions.append(true_label)
            else:
                predictions.append(perturbation_to_well_predictions[pert][0][0])
            labels.append(true_label)
        f1, precision, recall, accuracy = getScores(predictions=predictions, labels=labels, classes=classes)
        k_map[k] = ((f1, precision, recall, accuracy), (predictions, labels))
        if k == 1:
            print("    latent vote by wells, k={}: {} {}".format(k, accuracy, len((perturbation_to_well_predictions))))
    
    ##for each held-out perturbation, can also take the aggregate over all well-level embeddings of this perturbation, and find the class latent embedding that is closest to this aggregated embedding. Can do for top-k similarities.  
    ##find the aggregated embedding of each perturbation, key: perturbation, value: list of well-level embeddings that we will later take the aggregate over
    perturbation_to_aggregate = {}
    for i in range(0, len(holdout_latent["embeddings"])):
        if holdout_latent["perturbations"][i] not in perturbation_to_aggregate:
            perturbation_to_aggregate[holdout_latent["perturbations"][i]] = [holdout_latent["embeddings"][i]]
        else:
            perturbation_to_aggregate[holdout_latent["perturbations"][i]].append(holdout_latent["embeddings"][i])
    if class_aggregator == "median":
        perturbation_to_aggregate = {key: np.median(np.array(perturbation_to_aggregate[key]), axis=0) for key in perturbation_to_aggregate}
    if class_aggregator == "pca":
        perturbation_to_aggregate = {key: PCAAggregate(np.array(perturbation_to_aggregate[key])) for key in perturbation_to_aggregate}
    predictions, labels = [], [] ##predictions will be a list of sorted (class, metric score) (descending by metric score)
    for perturbation in perturbation_to_aggregate:
        true_label = perturbation_to_label[perturbation]
        highest_score = 0
        pred_class = ""
        all_class_preds = []
        ##evaluate candidate aggregated embeddings and find best pred
        for class_label in class_to_aggregate_embedding:
            aggregate_embedding = class_to_aggregate_embedding[class_label]     
            if metric == "pearson":
                score = scipy.stats.pearsonr(perturbation_to_aggregate[perturbation], aggregate_embedding)[0]
            if metric == "cosine":
                score = 1.0 - scipy.spatial.distance.cosine(perturbation_to_aggregate[perturbation], aggregate_embedding)
            if metric == "distance":
                score = -1 * np.linalg.norm(perturbation_to_aggregate[perturbation] - aggregate_embedding) ##minimize positive distance = maximize negative distance
            all_class_preds.append((class_label, score))
        all_class_preds = sorted(all_class_preds, key=lambda x: x[1], reverse=True)
        predictions.append(all_class_preds)
        labels.append(true_label)
    latent_k_map = {} ##key: k, value: ((f1, precision, recall, accuracy), (sub_preds, sub_labels))
    for k in range(1, len(class_to_aggregate_embedding) + 1): ##go from top-1 to top-all (i.e. all classes possible, which should yield 100% -> look at enrichment)
        sub_preds, sub_labels = [], []
        for i in range(0, len(labels)):
            sub_labels.append(labels[i])
            top_k = predictions[i][0:k]
            top_k = [x[0] for x in top_k] ##get just MOAs and not (MOA, metric score) tuples
            if labels[i] in top_k:
                sub_preds.append(labels[i])
            else:
                sub_preds.append(top_k[0])
        f1, precision, recall, accuracy = getScores(predictions=sub_preds, labels=sub_labels, classes=classes)
        if k == 1:
            print("    assignment by {} embedding similarity, k={}: {} {}".format(class_aggregator, k, accuracy, len(sub_labels)))
        latent_k_map[k] = ((f1, precision, recall, accuracy), (sub_preds, sub_labels))
    return pred_labels_map, k_map, latent_k_map

def getScores(predictions=None, labels=None, classes=None):
    """
    Given non-binary predictions and labels, and classes as a list of MOAs, will return f1, precision, recall, and accuracy
    """
     ##binarize preds and labels
    binary_labels = sklearn.preprocessing.label_binarize(labels, classes=classes) ##classes arg needs to be the same order as output of the neural network 
    binary_predictions = sklearn.preprocessing.label_binarize(predictions, classes=classes) 
    f1 = sklearn.metrics.f1_score(binary_labels, binary_predictions, average="weighted", zero_division=0)
    precision = sklearn.metrics.precision_score(binary_labels, binary_predictions, average="weighted", zero_division=0)
    recall = sklearn.metrics.recall_score(binary_labels, binary_predictions, average="weighted", zero_division=0)
    accuracy = sklearn.metrics.accuracy_score(binary_labels, binary_predictions)
    return f1, precision, recall, accuracy

def getHoldoutCompoundPrediction(model=None, study=None, loader=None, label_index_map=None, label_type=None):
    """
    Loops over test set, will assign a perturbation's MOA by the majority vote of the predictions over all its 1) images and 2) wells
    """
    perturbation_to_prediction = {} ##key perturbation, value: list of all predicted classes for each image field
    perturbation_to_label = {} ##key perturbation, value: actual MOA
    perturbation_well_prediction = {} ##perturbation: plate_well: list of predictions, condense list, then condense wells
    reverse_map = {value: key for (key, value) in label_index_map.items()}
    classes = []
    for i in range(0, len(reverse_map)):
        classes.append(reverse_map[i])
    model.eval()
    with torch.no_grad():
        for batch_idx, (imagename, input, target, perturbation) in enumerate(loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            _, pred = output.data.cpu().topk(1, dim=1) ##output = batch x # classes, pred = indices of max element for each instance            
            batch_preds = pred.squeeze().tolist()
            batch_preds = [reverse_map[p] for p in batch_preds]
            batch_labels = target.tolist()
            batch_labels = [reverse_map[l] for l in batch_labels]
            for i in range(0, len(perturbation)):
                ##get labels map
                if perturbation[i] not in perturbation_to_label:
                    perturbation_to_label[perturbation[i]] = batch_labels[i]
                else:
                    assert(perturbation_to_label[perturbation[i]] == batch_labels[i])
                ##get preds map 
                if perturbation[i] not in perturbation_to_prediction:
                    perturbation_to_prediction[perturbation[i]] = [batch_preds[i]]
                else:
                    perturbation_to_prediction[perturbation[i]].append(batch_preds[i])
                ##get more granular preds map 
                plate_well = getBarcode(imagename[i]) + getRowColumn(imagename[i])
                if perturbation[i] not in perturbation_well_prediction:
                    perturbation_well_prediction[perturbation[i]] = {plate_well: [batch_preds[i]]}
                else:
                    if plate_well not in perturbation_well_prediction[perturbation[i]]:
                        perturbation_well_prediction[perturbation[i]][plate_well] = [batch_preds[i]]
                    else:
                        perturbation_well_prediction[perturbation[i]][plate_well].append(batch_preds[i])
    perturbation_to_cousins = {} ##key: perturbation, value: (MOA, [other perturbations that affect the same MOA])
    test_perturbations = list(perturbation_well_prediction.keys())
    moa_to_compound = pickle.load(open("pickles/{}/poly_compound_moas_map.pkl".format(study), "rb"))
    compound_to_moa = pickle.load(open("pickles/{}/compound_to_moa_map.pkl".format(study), "rb"))
    for perturbation in test_perturbations:
        moa = compound_to_moa[perturbation]
        assert(len(moa) == 1)
        moa = moa.pop()
        compounds = moa_to_compound[moa]
        perturbation_to_cousins[perturbation] = (moa, compounds)

    ##analysis where we take the MOA as the majority MOA among image fields:
    perturbation_to_prediction = {key: Counter(perturbation_to_prediction[key]).most_common() for key in perturbation_to_prediction} ##key: perturbation, value: Counter object [(most frequent class, count), (second most frequent class, count) ... ]
    img_k_map  = {} #key: k, value: (accuracy, precision, recall)
    for k in range(1, 10): ##if we loosen the prediction-label match such that a correct prediction occurs if the label occurs in the top-k most frequent MOAs, k=1 is exact match
        predictions, labels, perts = [], [], []
        for perturbation in perturbation_to_prediction:
            most_common = perturbation_to_prediction[perturbation] 
            top_k = getTopKIncludingTies(most_common, k)
            if perturbation_to_label[perturbation] in top_k: ##if label is in the top-k MOAs take it as the pred
                predictions.append(perturbation_to_label[perturbation])
            else:
                ##if pred is wrong and there's a tie for top-1, let's choose the first alphabetically
                top_one = sorted(getTopKIncludingTies(most_common, 1))
                predictions.append(top_one[0])
            labels.append(perturbation_to_label[perturbation])
            perts.append(perturbation)
        f1, precision, recall, accuracy = getScores(predictions=predictions, labels=labels, classes=classes)
        img_k_map[k] = (accuracy, precision, recall)
        print("vote by images {}: {}, {}, {}".format(k, accuracy, precision, recall))
        print(len(set(labels)), len(perturbation_to_prediction))
        ##get list of correct compounds
        if k == 1:
            corrects = []
            for i in range(0, len(predictions)):
                if predictions[i] == labels[i]:
                    corrects.append(perts[i])
            for pert in corrects:
                print(pert, perturbation_to_cousins[pert])
    
    ##analysis where we take the MOA as the majority MOA among wells
    for perturbation in perturbation_well_prediction:
        true_label = perturbation_to_label[perturbation]
        ##condense list of field predictions
        for plate_well in perturbation_well_prediction[perturbation]:
            field_count = Counter(perturbation_well_prediction[perturbation][plate_well]).most_common()
            top_one = getTopKIncludingTies(field_count, k=1) 
            ## top_one can be multiple items if tied, if so and one of them is right, go with the true label 
            if true_label in top_one:
                perturbation_well_prediction[perturbation][plate_well] = true_label
            else:
                ##if pred is wrong and there's a tie for top one, let's choose the first alphabetically -- arbitrary but needed for consistency 
                top_one = sorted(top_one)
                perturbation_well_prediction[perturbation][plate_well] = top_one[0]    
    field_k_map  = {} #key: k, value: (accuracy, precision, recall)
    largest_k = 24 if study == "JUMP1" else 11 ##JUMP1 had vast majority 23 compound replicates, LINCS had a range from 0 -> 10 
    for k in range(1, largest_k):
        predictions, labels, perts = [], [], []
        for perturbation in perturbation_well_prediction:
            true_label = perturbation_to_label[perturbation]
            well_preds = list(perturbation_well_prediction[perturbation].values())
            most_common = Counter(well_preds).most_common() 
            top_k = getTopKIncludingTies(most_common, k)
            if true_label in top_k: ##if label is one of the top-k most frequent MOAs, take it as prediction
                predictions.append(true_label)
            else: ##assign prediction as one of the most frequent MOA, which of the wrong ones we pick doesn't matter for consistency 
                predictions.append(most_common[0][0]) 
            labels.append(true_label)
            perts.append(perturbation)
        ##binarize preds and labels 
        f1, precision, recall, accuracy = getScores(predictions=predictions, labels=labels, classes=classes)
        field_k_map[k] = (accuracy, precision, recall)
        print("vote by wells k={}: {}, {}, {}".format(k, accuracy, precision, recall))
        print(len(set(labels)), len(perturbation_well_prediction))
        ##for exact match preds / labels do further analysis 
        if k == 1:
            ##get list of correct compounds
            corrects = []
            for i in range(0, len(predictions)):
                if predictions[i] == labels[i]:
                    corrects.append(perts[i])
            for pert in corrects:
                print("    ", pert, perturbation_to_cousins[pert])
            ##for each MOA cardinality, find the specific performance i.e. what is the specific performance for compounds that have a MOA family size of 2 compounds including myself?
            max_cardinality = max([len(perturbation_to_cousins[pert][1]) for pert in perturbation_to_cousins])
            for cardinality in range(2, max_cardinality + 1):
                indices = [j for j in range(0, len(perts)) if len(perturbation_to_cousins[perts[j]][1]) == cardinality]
                if len(indices) == 0: ##if no perturbations have this cardinality, continue
                    continue
                subset_perts = [perts[i] for i in indices]
                subset_labels = [labels[i] for i in indices]
                subset_predictions = [predictions[i] for i in indices]
                f1, precision, recall, accuracy = getScores(predictions=subset_predictions, labels=subset_labels, classes=classes)
                print("cardinality {}, n={} compounds: {}, {}, {} ".format(cardinality, len(set(subset_perts)), accuracy, precision, recall))
    return img_k_map, field_k_map, corrects

def getTopKIncludingTies(l, k):
    """
    Gets the top k elements of l such that ties are both included
    Can think of it like get all elements within l such that we have the top-k unique counts in our return list:
        # l = (M1, 3), (M2, 3) (M3, 2) (M4,1) ; k = 3
        # -> (M1, 3), (M2, 3) (M3, 2) (M4,1)  
    """
    sorted_unique_counts = sorted(list(set([element[1] for element in l])), reverse=True)
    top_k_unique_counts = sorted_unique_counts[0: k] ##if k > sorted_unique_counts length, will take the entire list
    return_l = [element[0] for element in l if element[1] in top_k_unique_counts]
    return return_l

def standardizeEmbeddings(latent_dictionary):
    """
    Given a latent dictionary with key "embeddings", will standardize to zero mean and unit variance for each column
    """
    latent_copy = copy.deepcopy(latent_dictionary)
    matrix = []
    for i in range(0, len(latent_copy["embeddings"])):
        matrix.append(latent_copy["embeddings"][i])
    matrix = np.array(matrix)
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + .0000000000001) ##add small offset in case std is 0
    for i in range(0, len(latent_copy["embeddings"])):
        latent_copy["embeddings"][i] = matrix[i]
    return latent_copy

def standardizeEmbeddingsByDMSO(latent_dictionary):
    latent_copy = copy.deepcopy(latent_dictionary)
    DMSO_matrix = []
    full_matrix = []
    for i in range(0, len(latent_copy["embeddings"])):
        if latent_copy["perturbations"][i] in ["DMSO", "Empty"]:
            DMSO_matrix.append(latent_copy["embeddings"][i])
        full_matrix.append(latent_copy["embeddings"][i])
    DMSO_matrix = np.array(DMSO_matrix)
    full_matrix = np.array(full_matrix)
    full_matrix = (full_matrix - DMSO_matrix.mean(axis=0)) / (DMSO_matrix.std(axis=0) + .0000000000001) ##add small offset in case std is 0
    for i in range(0, len(latent_copy["embeddings"])):
        latent_copy["embeddings"][i] = full_matrix[i]
    return latent_copy

def standardizeEmbeddingsByDMSOPlate(latent_dictionary):
    latent_copy = copy.deepcopy(latent_dictionary)
    matrix = []
    plate_to_DMSO_embeddings = {plate: [] for plate in set([getBarcode(latent_copy["wells"][i]) for i in range(0, len(latent_copy["wells"]))])} ##key plate, value: list of DMSO embeddings for that plate
    plate_to_other_embeddings = {plate: [] for plate in set([getBarcode(latent_copy["wells"][i]) for i in range(0, len(latent_copy["wells"]))])}##key plate, value: list of other embeddings for that plate
    DMSO_plate_stats_map = {} ##key plate, value: DMSO [mean, std]
    ##instantiate plate to embeddings
    for i in range(0, len(latent_copy["embeddings"])):
        plate = getBarcode(latent_copy["wells"][i])
        if latent_copy["perturbations"][i] in ["DMSO", "Empty"]:
            plate_to_DMSO_embeddings[plate].append(latent_copy["embeddings"][i])
        else:
            plate_to_other_embeddings[plate].append(latent_copy["embeddings"][i])
    ##calculate mean and std of DMSO per plate 
    for plate in plate_to_DMSO_embeddings:
        DMSO_plate_stats_map[plate] = np.mean(np.array(plate_to_DMSO_embeddings[plate]), axis=0), np.std(np.array(plate_to_DMSO_embeddings[plate]), axis=0)
    ##standardize embeddings by DMSO
    new_embeddings = []
    for i in range(0, len(latent_copy["embeddings"])):
        plate = getBarcode(latent_copy["wells"][i])
        mean, std = DMSO_plate_stats_map[plate]
        new_embeddings.append((latent_copy["embeddings"][i] - mean) / (std + .0000000000001))
    latent_copy["embeddings"] = new_embeddings
    return latent_copy

def standardizeEmbeddingsByPlate(latent_dictionary):
    """
    Given a latent dictionary with key "embeddings", will standardize to zero mean and unit variance for each column by plate
    """
    latent_copy = copy.deepcopy(latent_dictionary)
    matrix = []
    plate_to_embeddings = {} ##key plate, value: list of embeddings for that plate
    plate_stats_map = {} ##key plate, value: [mean, std]
    for i in range(0, len(latent_copy["embeddings"])):
        plate = getBarcode(latent_copy["wells"][i])
        if plate not in plate_to_embeddings:
            plate_to_embeddings[plate] = [latent_copy["embeddings"][i]]
        else:
            plate_to_embeddings[plate].append(latent_copy["embeddings"][i])
    for plate in plate_to_embeddings:
        plate_stats_map[plate] = np.mean(np.array(plate_to_embeddings[plate]), axis=0), np.std(np.array(plate_to_embeddings[plate]), axis=0)
    new_embeddings = []
    for i in range(0, len(latent_copy["embeddings"])):
        plate = getBarcode(latent_copy["wells"][i])
        mean, std = plate_stats_map[plate]
        new_embeddings.append((latent_copy["embeddings"][i] - mean) / (std + .0000000000001))
    latent_copy["embeddings"] = new_embeddings
    return latent_copy

def filterToTestSet(latent_dictionary, test_csv, study=None):
    test_df = pd.read_csv(test_csv)
    if study == "JUMP1":
        test_wells = set([getJumpBatch(test_df["imagename"][i]) + getBarcode(test_df["imagename"][i]) + getRowColumn(test_df["imagename"][i]) for i in range(0, len(test_df))])
    if study == "lincs":
        barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb"))
        test_wells =  set([barcode_to_platemap[getBarcode(test_df["imagename"][i])] + getBarcode(test_df["imagename"][i]) + getRowColumn(test_df["imagename"][i]) for i in range(0, len(test_df))])
    latent_copy = copy.deepcopy(latent_dictionary)
    indices_to_keep = []
    for i in range(0, len(latent_copy["wells"])):
        if latent_copy["wells"][i] in test_wells:
            indices_to_keep.append(i)
    for key in latent_copy:
        latent_copy[key] = [latent_copy[key][i] for i in indices_to_keep]
    return latent_copy

def logisticRegression(latent_dictionary, study=None, training_csv=None, test_csv=None, drop_neg_control=True):
    if drop_neg_control:
        latent_copy = removeEmptyLabels(latent_dictionary) 
    latent_copy = copy.deepcopy(latent_dictionary)
    training_df = pd.read_csv(training_csv)
    test_df = pd.read_csv(test_csv)
    if study == "JUMP1":
        training_wells = set([getJumpBatch(training_df["imagename"][i]) + getBarcode(training_df["imagename"][i]) + getRowColumn(training_df["imagename"][i]) for i in range(0, len(training_df))])
        test_wells =  set([getJumpBatch(test_df["imagename"][i]) + getBarcode(test_df["imagename"][i]) + getRowColumn(test_df["imagename"][i]) for i in range(0, len(test_df))])
    if study == "lincs":
        barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb"))
        training_wells = set([barcode_to_platemap[getBarcode(training_df["imagename"][i])] + getBarcode(training_df["imagename"][i]) + getRowColumn(training_df["imagename"][i]) for i in range(0, len(training_df))])
        test_wells =  set([barcode_to_platemap[getBarcode(training_df["imagename"][i])] + getBarcode(test_df["imagename"][i]) + getRowColumn(test_df["imagename"][i]) for i in range(0, len(test_df))])
    ##partition latent_dictionary into training and test
    training_latent = {"labels": [], "embeddings":[], "perturbations":[]}
    test_latent = {"labels": [], "embeddings":[], "perturbations":[]}
    validation_or_not_found = []
    for i in range(0, len(latent_copy["perturbations"])):
        if latent_copy["wells"][i] in training_wells:
            training_latent["labels"].append(latent_copy["labels"][i])
            training_latent["perturbations"].append(latent_copy["perturbations"][i])
            training_latent["embeddings"].append(latent_copy["embeddings"][i])
        elif latent_copy["wells"][i] in test_wells:
            test_latent["labels"].append(latent_copy["labels"][i])
            test_latent["perturbations"].append(latent_copy["perturbations"][i])
            test_latent["embeddings"].append(latent_copy["embeddings"][i])
        else:
            validation_or_not_found.append(latent_copy["wells"][i])
    ##remove single compound MOAs from test latent
    test_latent = removeSingleCompoundMOAEmbeddings(test_latent)
    training_embeddings = np.array(training_latent["embeddings"])
    training_labels = list(training_latent["labels"])
    test_embeddings = np.array(test_latent["embeddings"])
    test_labels = list(test_latent["labels"])
    logisticRegr = LogisticRegression(max_iter=10000)   
    logisticRegr.fit(training_embeddings, training_labels)
    test_predictions = logisticRegr.predict(test_embeddings)
    scores = getScores(predictions=test_predictions, labels=test_labels, classes=list(set(latent_copy["labels"])))
    scores = [scores[0], scores[1], scores[2]] ##f1, precision, recall
    return scores

def removeSingleCompoundMOAEmbeddings(latent_dictionary):
    """
    Will remove all embeddings that belong to an MOA that are just represented by one compound (cousin-less compounds) EXCEPT DMSO control
    and return a copy
    """
    latent_copy = copy.deepcopy(latent_dictionary)
    moa_to_perturbations = {m: set() for m in set(latent_dictionary["labels"])} ##key: moa, value: set of perturbations
    for i in range(0, len(latent_dictionary["labels"])):
        moa_to_perturbations[latent_dictionary["labels"][i]].add(latent_dictionary["perturbations"][i])
    single_compound_moas = [m for m in moa_to_perturbations if len(moa_to_perturbations[m]) < 2]
    indices_to_keep = []
    for i in range(0, len(latent_copy["labels"])):
        if latent_copy["labels"][i] not in single_compound_moas or latent_copy["perturbations"][i] in ["DMSO", "Empty"]:
            indices_to_keep.append(i)
    for key in latent_copy:
        latent_copy[key] = [latent_copy[key][i] for i in indices_to_keep]
    return latent_copy

def removeSingleCompoundMOAEmbeddingsFromDF(df):
    """
    From Pandas Dataframe, will remove all embeddings that belong to an MOA that are just represented by one compound (cousin-less compounds) EXCEPT DMSO control
    and return a copy
    """
    df_copy = df.copy()
    moa_to_perturbations = {m: set() for m in set(df_copy["moas"])} ##key: moa, value: set of perturbations
    for index, row in df.iterrows():
        moa_to_perturbations[row["moas"]].add(latent_dictionary["perturbations"])
    single_compound_moas = [m for m in moa_to_perturbations if len(moa_to_perturbations[m]) < 2]
    df_copy = df_copy[~df_copy["moas"].isin(single_compound_moas)]
    return df_copy
