
import os

import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import openpyxl
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import openpyxl

import src.trainer2 as trainer2
import src.models as models
import src.losses as losses
import src.metrics as metrics
import src.transforms as transforms
import src.dataset as Dataset


def get_paths_to_patient_files(path_to_imgs, path_results):
    patients = os.listdir(path_to_imgs)
    patient_num = int(len(patients))
    paths = []

    for i in range(patient_num):
        file_list = os.listdir( path_to_imgs + '/' + str(patients[i]))


        path_to_ct = path_to_imgs + '/' + str(patients[i]) +'/CT.nii.gz'
        path_to_pet = path_to_imgs + '/' + str(patients[i]) + '/PET.nii.gz'
        path_to_seg = path_to_imgs + '/' + str(patients[i]) + '/SEG.nii.gz'
        path_to_output = path_results + '/' + str(patients[i]) + '_output.nii.gz'
        paths.append((path_to_pet, path_to_ct, path_to_seg, path_to_output))

    return paths


def main():
    torch.multiprocessing.freeze_support()

    path_to_train_data = 'C:/Users/joming/PycharmProjects/Wholebody_leison/preprocessing_dataset/lesion'
    path_to_val_data = 'C:/Users/joming/PycharmProjects/Wholebody_leison/preprocessing_dataset/valdation'
    path_to_out_data = './train_result_image'
    path_to_val_out = './val_result_image'
    path_to_save_dir = './result_이어서'


    filepath = os.path.join(path_to_save_dir, 'results.xlsx')
    wb = openpyxl.Workbook()
    wb.save(filepath)


    train_batch_size = 1
    val_batch_size = 1
    num_workers = 0
    lr = 1e-3  # initial learning rate
    n_epochs = 150
    n_cls = 2  # number of classes to predict (background and tumor)
    in_channels = 3  # number of input modalities
    n_filters = 16  # number of filters after the input
    reduction = 2  # parameter controls the size of the bottleneck in SENorm layers
    T_0 = 25  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
    eta_min = 1e-5


    train_paths = get_paths_to_patient_files(path_to_train_data,path_to_out_data)
    val_paths = get_paths_to_patient_files(path_to_val_data,path_to_val_out)

    train_transforms = transforms.Compose([
        # transforms.Mirroring(p=0.5),
        # transforms.RandomRotation_custom(p=0.5, angle_range=[0, 45]),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])
    #
    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])

    # datasets:
    train_set = Dataset.HecktorDataset(train_paths, transforms=train_transforms)
    val_set = Dataset.HecktorDataset(val_paths, transforms=val_transforms)
    #
    #dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)


    dataloaders = {
        'train': train_loader,
        'val': val_loader}

    model = models.BaselineUNet_down3_DAM(in_channels, n_cls, n_filters)
    model.load_state_dict(torch.load(os.path.join('./result', '19_weights.pt')))
    criterion = losses.Dice_and_FocalLoss()
    fp1_loss = losses.FocalLoss()
    fp2_loss = losses.FocalLoss()
    fp3_loss = losses.FocalLoss()
    fp4_loss = losses.FocalLoss()
    fn1_loss = losses.FocalLoss()
    fn2_loss = losses.FocalLoss()
    fn3_loss = losses.FocalLoss()
    fn4_loss = losses.FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metric = metrics.dice
    recall = metrics.recall
    precision = metrics.precision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    trainer_ = trainer2.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        fp1_loss=fp1_loss,
        fp2_loss=fp2_loss,
        fp3_loss=fp3_loss,
        fp4_loss=fp4_loss,
        fn1_loss=fn1_loss,
        fn2_loss=fn2_loss,
        fn3_loss=fn3_loss,
        fn4_loss=fn4_loss,
        optimizer=optimizer,
        metric=metric,
        recall=recall,
        precision=precision,
        scheduler=scheduler,
        num_epochs=n_epochs,
        parallel=True,
        filepath=filepath)

    trainer_.train_model(path_to_dir=path_to_save_dir)
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ =='__main__':
    main()
