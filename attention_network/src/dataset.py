import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pathlib
import skimage.transform as skTrans
import matplotlib.pyplot as plt

def read_data(path_to_nifti, return_numpy=True):
    """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))

def read_data_header(path_to_nifti):
    """Read a header file of Nifti image"""
    return nib.load(str(path_to_nifti))



class HecktorDataset(Dataset):
    def __init__(self, sample_path, transforms=None):
        self.sample_path = sample_path
        self.transforms = transforms
        #index: 0~99
        #rand[index]: 100 random sampling of total 513
        #total:513

    def __len__(self):
        return len(self.sample_path)

    def __getitem__(self, index):
        sample = dict()



        header = read_data_header(self.sample_path[index][-1])
        sample['header'] = header.affine

        sample['id'] = str(self.sample_path[index][0]).split('/')[-2]

        PET = read_data(self.sample_path[index][0])
        CT = read_data(self.sample_path[index][1])

        sample['shape'] = PET.shape



        # resize_pet = skTrans.resize(PET, (192,192,192), order=2, preserve_range=True)
        # resize_ct = skTrans.resize(CT, (192,192,192), order=2, preserve_range=True)

        # img = [resize_pet, resize_ct]
        img = [PET, CT]
        img = np.stack(img, axis=-1)
        sample['input'] = img

        mask = read_data(self.sample_path[index][-1])
        mask = np.expand_dims(mask, axis=3)
        assert img.shape[:-1] == mask.shape[:-1]
        sample['target'] = mask



        if self.transforms:
            sample = self.transforms(sample)



        return sample



# class HecktorDataset(Dataset):
#     def __init__(self, sample_path):
#         self.sample_path = sample_path
#
#
#     def __len__(self):
#         return len(self.sample_path)
#
#     def normalize_ct(self,img):
#         norm_img = np.clip(img, -1024, 1024) / 1024
#         return norm_img
#
#     def __getitem__(self, index):
#         sample = dict()
#
#
#         header = read_data_header(self.sample_path[index][-1])
#         sample['header'] = header.header['pixdim']
#         sample['id'] = str(self.sample_path[index][0]).split('/')[-2]
#
#         ct = read_data(self.sample_path[index][1])
#         ct = self.normalize_ct(ct)
#         pet = read_data(self.sample_path[index][0])
#
#
#         img = [pet,ct] #C H W D
#         img = np.array(img)
#         img = img.transpose((0,3,1,2)) #C D H W
#         sample['image'] = torch.from_numpy(img).float()
#
#         mask = [read_data(self.sample_path[index][-1])]
#         mask = np.array(mask)
#         mask = mask.transpose((0,3,1,2))
#
#
#         assert img.shape[1:] == mask.shape[1:]
#         sample['label'] = torch.from_numpy(mask).float()
#
#
#         image_copy = np.zeros(img.shape).astype(np.float64)
#         image_copy[:,1:,:,:] = img[:,0:img.shape[1]-1,:,:]
#         image_res = img - image_copy
#         image_res[:,0,:,:] = 0
#         image_res = np.abs(image_res)
#         sample['image_res'] = torch.from_numpy(image_res).float()
#
#         label_copy = np.zeros(mask.shape).astype(np.float64)
#         label_copy[:,1:,:,:] = mask[:,0:mask.shape[1]-1,:,:]
#         label_res = mask-label_copy
#         label_res[np.where(label_res == 0)] = 0
#         label_res[np.where(label_res != 0)] = 1
#         sample['label_res'] = torch.from_numpy(label_res).float()
#
#         return sample
#
#
# class HecktorDataset2(Dataset):
#     def __init__(self, sample_path):
#         self.sample_path = sample_path
#
#
#     def __len__(self):
#         return len(self.sample_path)
#
#     def normalize_ct(self,img):
#         norm_img = np.clip(img, -1024, 1024) / 1024
#         return norm_img
#
#     def normalize_pt(self,img):
#         mean = np.mean(img)
#         std = np.std(img)
#         return (img - mean) / (std + 1e-3)
#
#     def __getitem__(self, index):
#         sample = dict()
#
#
#         header = read_data_header(self.sample_path[index][-1])
#         sample['header'] = header.header['pixdim']
#         sample['id'] = str(self.sample_path[index][0]).split('/')[-2]
#
#         ct = read_data(self.sample_path[index][1])
#         ct = self.normalize_ct(ct)
#         pet = read_data(self.sample_path[index][0])
#         pet = self.normalize_pt(pet)
#
#
#         img = [pet,ct] #C H W D
#         img = np.array(img)
#         img = img.transpose((0,3,1,2)) #C D H W
#         sample['image'] = torch.from_numpy(img).float()
#
#         mask = [read_data(self.sample_path[index][-1])]
#         mask = np.array(mask)
#         mask = mask.transpose((0,3,1,2))
#
#
#         assert img.shape[1:] == mask.shape[1:]
#         sample['label'] = torch.from_numpy(mask).float()
#
#
#         image_copy = np.zeros(img.shape).astype(np.float64)
#         image_copy[:,1:,:,:] = img[:,0:img.shape[1]-1,:,:]
#         image_res = img - image_copy
#         image_res[:,0,:,:] = 0
#         image_res = np.abs(image_res)
#         sample['image_res'] = torch.from_numpy(image_res).float()
#
#         label_copy = np.zeros(mask.shape).astype(np.float64)
#         label_copy[:,1:,:,:] = mask[:,0:mask.shape[1]-1,:,:]
#         label_res = mask-label_copy
#         label_res[np.where(label_res == 0)] = 0
#         label_res[np.where(label_res != 0)] = 1
#         sample['label_res'] = torch.from_numpy(label_res).float()
#
#         return sample