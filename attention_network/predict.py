import os
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
import numpy as np
import nibabel as nib

import src.trainer2 as trainer2
import src.models as models
import src.losses as losses
import src.metrics as metrics
import src.transforms as transforms
import src.dataset as Dataset







path_to_test_data = 'C:/Users/joming/PycharmProjects/Wholebody_leison/preprocessing_dataset/valdation'
path_to_save_dir = './results'

n_cls = 2  # number of classes to predict (background and tumor)
in_channels = 2 # number of input modalities
n_filters = 16

cuda_device = "cuda:0"
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    print('Start training the model on CPU')
else:
    print(f'Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 4, 1)
fn_class = lambda x: 1.0 * (x > 0.5)



def get_paths_to_patient_files(path_to_imgs):
    patients = os.listdir(path_to_imgs)
    patient_num = int(len(patients))
    paths = []

    for i in range(patient_num):
        file_list = os.listdir( path_to_imgs + '/' + str(patients[i]))


        path_to_ct = path_to_imgs + '/' + str(patients[i]) +'/CT.nii.gz'
        path_to_pet = path_to_imgs + '/' + str(patients[i]) + '/PET.nii.gz'
        path_to_seg = path_to_imgs + '/' + str(patients[i]) + '/SEG.nii.gz'
        paths.append((path_to_pet,path_to_ct,path_to_seg))

    return paths

test_paths = get_paths_to_patient_files(path_to_test_data)

test_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])

test_set = Dataset.HecktorDataset(test_paths,transforms=test_transforms)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


criterion = losses.DiceLoss()
metric = metrics.dice
recall = metrics.recall
precision = metrics.precision

model = models.MSA_ITN(in_channels, n_cls, n_filters).to(device)
model.load_state_dict(torch.load(os.path.join(path_to_save_dir,'best_model_weights.pt')))
model.eval()

phase_loss = 0.0  # Train or val loss
phase_metric = 0.0
phase_recall = 0.0
phase_precision = 0.0

result_txt = os.path.join(path_to_save_dir,'val_result.txt')
if os.path.isfile(result_txt):
    os.unlink(result_txt)
if not os.path.isfile(result_txt):
    f = open(result_txt, 'w')
    f.close()

result_path = os.path.join(path_to_save_dir,'val_result_image')
if os.path.isdir(result_path):
    os.rmdir(result_path)
if not os.path.isdir(result_path):
    os.mkdir(result_path)

with torch.no_grad():
    for data in test_loader:
        # forward pass
        header = data['header']
        header = header.detach().cpu().numpy()
        header = header[0,:,:]
        print(header.shape)
        id = data['id']
        input, target = data['input'], data['target']
        input, target = input.to(device), target.to(device)

        output = model(input)
        output_threshold = fn_class(output)


        loss_ = criterion(output_threshold, target)
        dice_ = metric(output.detach(), target.detach())
        recall_ = recall(output.detach(), target.detach())
        precision_ = precision(output.detach(),target.detach())

        phase_loss += loss_.item()
        phase_metric += dice_.item()
        phase_recall += recall_.item()
        phase_precision += precision_.item()

        with open(result_txt, 'a') as f:
            data = f'{id[0]} \tloss: \t{loss_:.5f} \tdice: \t{dice_:.5f} \trecall: \t{recall_:.5f} \tprecision: \t{precision_:.5f} \n'
            f.write(data)

        output = np.squeeze(fn_tonumpy(output))
        # 사진 결과 저장하기 (thresholing한 것)
        # proxy = nib.load('C:/Users/joming/PycharmProjects/Wholebody_leison/preprocessing_dataset/val/' + id[0] + '/SEG.nii.gz')

        # img_pt = nib.Nifti1Image(input_pt, proxy.affine, proxy.header)
        # img_ct = nib.Nifti1Image(input_ct, proxy.affine, proxy.header)
        img_output = nib.Nifti1Image(output, header)
        # img_attn = nib.Nifti1Image(attn,np.eye(4))
        #
        # img_ct.to_filename(os.path.join(result_path, id + '_ct.nii.gz'))
        # img_pt.to_filename(os.path.join(result_path, id + '_pt.nii.gz'))
        img_output.to_filename(os.path.join(result_path, id[0] + '_output.nii.gz'))
        # img_attn.to_filename(os.path.join(result_path, id + '_attn.nii.gz'))

phase_loss /= len(test_loader)
phase_metric /= len(test_loader)
phase_recall /= len(test_loader)
phase_precision /= len(test_loader)

with open(result_txt, 'a') as f:
    data = f'Test loss: \t{phase_loss:.5f} \tdice: \t{phase_metric:.5f} \trecall: \t{phase_recall:.5f} \t:precision \t{phase_precision:.5f}'
    f.write(data)
