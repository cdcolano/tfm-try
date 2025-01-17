import os
import cv2 
import math, json, glob
import numpy as np
from io import BytesIO
from PIL import Image
from itertools import cycle
import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py
import albumentations as A
import traceback
from torch.utils.data._utils.collate import default_collate
from skimage.metrics import structural_similarity as ssim

def remove_background(image_path):
    input_image = Image.open(image_path)
    #output_image = remove(input_image)
    return input_image.convert('L') 

def compute_similarity(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ssim(img1, img2)

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds

def read_json_KRT(fname):
    with open(fname,'r') as f:
        data = json.load(f)

    KRT = torch.Tensor((np.array(data['K'])@np.array(data['RT'][:3])))
    return KRT

def get_transform(size=256, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    
    
    transform_list.append(transforms.Resize(size, interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class dataloader(Dataset):

    def __init__(self, args, paths, samples_list, num_mesh_images = 5, mode = 'train'):
        good_samples_list, bad_samples_list = samples_list
        self.good_folder_path, self.bad_folder_path = paths
        self.mode = mode
        self.args = args
        self.num_mesh_images = num_mesh_images
        self.n_pnts = args.n_pnts

        size = 256
        self.transform_box = A.Compose([
            A.Resize(size, size),
            A.RandomCrop(width=224, height=224, p = 1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(size, size),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.transform_test = A.Compose([
            A.Resize(size, size),
            A.RandomCrop(width=224, height=224, p = 0.2),
            A.HorizontalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(size, size),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        split_dict = {'train': [i['anno_id'] for i in json.load(open('data/splits/train_ids.json','r'))],
                    'test' : [i['anno_id'] for i in json.load(open('data/splits/test_ids.json','r'))],
                    'val' :  [i['anno_id'] for i in json.load(open('data/splits/val_ids.json','r'))]}
        self.type_dict = {'position':1, 'rotate':2, 'missing':3, 'damaged':4, 'swapped':5}

        good_train_list, good_test_list, good_val_list = [], [], []
        bad_train_list, bad_test_list, bad_val_list = [], [], []

        for i in good_samples_list:
            if i.split('/')[-3] in split_dict['test']: #TODO NO FUNCIONA PORQUE EL SPLIT NO ES POR / EN WINWOS HABRIA QUE USAR EL FILE  SEPARATOR
                good_test_list.append(i)
            elif i.split('/')[-3] in split_dict['val']:
                good_val_list.append(i)
            else:
                good_train_list.append(i)

        for i in bad_samples_list:
            #if i.split('/')[-3].split('_')[0][1:] in split_dict['test']:
            if i.split('/')[-3] in split_dict['test']:
                bad_test_list.append(i)
            #elif i.split('/')[-3].split('_')[0][1:] in split_dict['val']:
            elif i.split('/')[-3] in split_dict['val']:
                bad_val_list.append(i)
            else:
                bad_train_list.append(i)

        if mode == 'train':
            self.good_list, self.bad_list = good_train_list, bad_train_list
            self.good_list_upsampled = self.good_list
        elif mode == 'test':
            self.good_list, self.bad_list = good_test_list, bad_test_list
            self.good_list_upsampled = self.good_list
        elif mode == 'val':
            self.good_list, self.bad_list = good_val_list, bad_val_list
            self.good_list_upsampled = self.good_list
        
        
        self.paths = self.good_list_upsampled + self.bad_list
        self.labels = [0]*len(self.good_list_upsampled) + [1]*len(self.bad_list)

        # if self.args.pretraining: self.objs = glob.glob('/disk/scratch_ssd/s2514643/brokenchairs/normals/*')

        print (f'#dataset:[{mode}] total {len(self.good_list)+len(self.bad_list)} -> {len(self.good_list)} good and {len(self.bad_list)} bad images')


    def __len__(self):  
        
        return len(self.bad_list)+len(self.good_list_upsampled)

    def get_image_tensor(self, path):
        img = Image.open(path)
        trans = get_transform(size = 256)
        img = trans(img)
        return img    

    def get_mesh_image_tensor(self, path):
        img = Image.open(path).convert('RGB')
        trans = get_transform(size = 256)
        img = trans(img)
        return img    

    def to_image_tensor(self, input, aug = True):

        if not aug:
            if isinstance(input, list):
                img_path, bbox = input
            else:
                img_path = input
            imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            transformed = self.transform_test(image = imgs)
            imgs = transformed['image']
            imgs = transforms.ToTensor()(imgs)
            if isinstance(input, list):
                return imgs, bbox
            else:
                return imgs
                
        else:
            if isinstance(input, list):
                img_path, bbox = input
                imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                transformed = self.transform_box(image = imgs, bboxes = bbox.unsqueeze(0), class_labels = ['none'])
                imgs, bbox = transformed['image'], torch.Tensor(transformed['bboxes']).squeeze()
                imgs = transforms.ToTensor()(imgs)
                return [imgs, bbox]
            else:
                img_path = input
                imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                transformed = self.transform_box(image = imgs,  bboxes = [], class_labels = [])
                imgs = transformed['image']
                imgs = transforms.ToTensor()(imgs)
                return imgs

    def __getitem__(self, index):
        try:
            img_path = self.paths[index]
            #imgs = self.get_image_tensor(img_path)
            labels = self.labels[index]

            if labels == 1:
                bad_json_path = '/'.join(img_path.split('/')[:-2])+'/annotation/info_'+os.path.basename(img_path)[:-4]+'.json' #TODO REVISAR EL MENOS 4
                ref_mesh_path = glob.glob(f"{self.good_folder_path}/{img_path.split('/')[-3]}/rendered_mesh/*.png")
                #ref_mesh_sdf_path = f"{self.good_folder_path}/{img_path.split('/')[-3].split('_')[0][1:]}/sdf64x64.h5"
                bad_mask_path = '/'.join(img_path.split('/')[:-2])+'/annotation/mask_new_'+os.path.basename(img_path)[:-4]+'.png'
                gt_anomly_mask = transforms.ToTensor()(Image.open(bad_mask_path).resize((256,256)))
                with open(bad_json_path, 'r') as f:
                    info = json.load(f)
                    xi,yi,hi,wi = np.array(info['2d_bbox'])/512 #TODO NO ENTIENDO PORQUE ENTRE 512
                    xi = xi+hi/2
                    yi = yi+wi/2
                    bbox, anom_type = torch.Tensor([xi,yi,hi,wi]), self.type_dict[(info['config'])['type']]
                # imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                # transformed = self.transform_box(image = imgs, bboxes = bbox.unsqueeze(0), class_labels = ['none'])
                # imgs, bbox = transformed['image'], torch.Tensor(transformed['bboxes']).squeeze()
                # imgs = transforms.ToTensor()(imgs)
                imgs, bbox = self.to_image_tensor([img_path, bbox], aug = True)
                if bbox.shape[0] != 4: pass #print ('WRN: /data/error - ignoring...')
                assert bbox.shape[0] == 4
            else:
                gt_anomly_mask, bbox, anom_type = torch.zeros((1,256,256)), torch.Tensor([0,0,0,0]), 0
                ref_mesh_path = glob.glob(f"{self.good_folder_path}/{img_path.split('/')[-3]}/rendered_mesh/*.png")
                #ref_mesh_sdf_path = f"{self.good_folder_path}/{img_path.split('/')[-3]}/sdf64x64.h5"
                imgs = self.to_image_tensor(img_path, aug = True)
            c=0
            while True:
                random_2_views = np.random.choice(ref_mesh_path, 2, replace = False)
                color_img_path = str(np.random.choice(glob.glob(os.path.dirname(os.path.dirname(ref_mesh_path[0]))+'/renders/*.png')))
                xy_data_v1, xy_data_v2 = [np.load(i.replace('.png','.npy'), allow_pickle=True).item() for i in random_2_views]
                xy_data_c = np.load(color_img_path.replace('.png','.npy'), allow_pickle=True).item()
                paired_idx = np.where(xy_data_v1['is_visible'] & xy_data_v2['is_visible'] & xy_data_c['is_visible'])[0]

                c=c+1
                if len(paired_idx)>=self.n_pnts:
                    break
                if c>10:
                    return None
            pxy_v1, pxy_v2, pxy_c = xy_data_v1['pxy'][paired_idx], xy_data_v2['pxy'][paired_idx], xy_data_c['pxy'][paired_idx]
            n_far_indx = fps(pxy_v1, self.n_pnts)
            pxy_v1, pxy_v2, pxy_c = torch.Tensor(pxy_v1[n_far_indx]), torch.Tensor(pxy_v2[n_far_indx]), torch.Tensor(pxy_c[n_far_indx])

            img_v1 = self.get_mesh_image_tensor(random_2_views[0]) 
            img_v2 = self.get_mesh_image_tensor(random_2_views[1]) 
            img_c = self.to_image_tensor(color_img_path, aug = self.mode == 'train')
            img_query = remove_background(img_path)
            img_query_np = np.array(img_query)
            similarities = []
            for mesh_img_path in ref_mesh_path:
                mesh_img = Image.open(mesh_img_path) # Convert to grayscale
                mesh_img_np = np.array(mesh_img)
                sim = compute_similarity(img_query_np, mesh_img_np)
                similarities.append((mesh_img_path, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            num_mesh_images=min(self.num_mesh_images, len(ref_mesh_path))
            num_sim_images = num_mesh_images // 2
            num_random_images = num_mesh_images - num_sim_images
            top_similar_paths = [sim[0] for sim in similarities[:num_sim_images]]

            remaining_indices = np.arange(num_sim_images, len(ref_mesh_path))
            random_indices = np.random.choice(remaining_indices, num_random_images, replace=False)
            random_paths = [ref_mesh_path[i] for i in random_indices]
            selected_paths = top_similar_paths + random_paths
            #ref_mesh_path_select = [ref_mesh_path[i] for i in top_similar_indices]
            
            # selected_paths = set()
            # selected_cameras = set()
            
            
            # # Ensure unique camera selection in order of their SSIM scores
            # for mesh_img_path, sim in similarities:
            #     camera_id = mesh_img_path.split('_')[0]
            #     if camera_id not in selected_cameras:
            #         selected_paths.add(mesh_img_path)
            #         selected_cameras.add(camera_id)
            #     if len(selected_paths) >= self.num_mesh_images:
            #         break

            # # If limit is not reached, fill with remaining highest-scoring views
            # if len(selected_paths) < self.num_mesh_images:
            #     for mesh_img_path, sim in similarities:
            #         if mesh_img_path not in selected_paths:
            #             selected_paths.append(mesh_img_path)
            #             if len(selected_paths) >= self.num_mesh_images:
            #                 break
            ref_mesh_path_select=selected_paths
            #ref_mesh_path_select = np.random.choice(ref_mesh_path, self.num_mesh_images)
            mesh = torch.stack([self.get_mesh_image_tensor(i) for i in ref_mesh_path_select if i.endswith('.png')], 0)
            # if self.mode == 'train':
            #     num_images_to_select = min(self.num_mesh_images, len(ref_mesh_path))
            #     ref_mesh_path_select = np.random.choice(ref_mesh_path, num_images_to_select, replace=False)
            # else:
            #     num_images_to_select = min(self.num_mesh_images, len(ref_mesh_path))
            #     ref_mesh_path_select = np.random.choice(ref_mesh_path, num_images_to_select, replace=False)
            # mesh = torch.stack([self.get_mesh_image_tensor(i) for i in ref_mesh_path_select if i.endswith('.png')],0)
            cam_poses = []
            for i in ref_mesh_path_select:
                with open(i.replace('.png','.json'), 'r') as fi:
                    KRT = json.load(fi)
                    cam_pose_i =(np.array(KRT['K'])@np.array(KRT['RT'])[:3]).flatten()
                    cam_poses.append(cam_pose_i)
            cam_poses = torch.Tensor(cam_poses)

            pos_enc3d = torch.stack([torch.Tensor(np.load(i.replace('.png', '.npy'), allow_pickle = True).item()['xyz']) for i in ref_mesh_path_select])
            
            return_dict = {'img_v1':img_v1, 
                           'img_v2':img_v2, 
                           'pxy_v1':pxy_v1, 
                           'pxy_v2':pxy_v2, 
                           'pxy_c':pxy_c, 
                           'img_c':img_c, 
                           'imgs':imgs, 
                           'mesh':mesh,   
                           'labels':labels, 
                           'bbox':bbox, 
                           'gt_anomly_mask':gt_anomly_mask,
                           'anomaly_type': anom_type,
                           'cam_poses': cam_poses,
                           'pos_enc3d': pos_enc3d,
                           'path':img_path}
                           #'mesh_path': ref_mesh_path_select}
            
            # return_dict = {
            #                 'imgs':imgs, 
            #                 'path':img_path,
            #                 }
            return return_dict #img_v1, img_v2, pxy_v1, pxy_v2, color_img, imgs, mesh, labels, bbox, gt_anomly_mask #, anom_type, paths]

        except Exception as e:
            # Print the exception message for any other exceptions
            print(f"An error occurred in get_item : {e}")
            traceback.print_exc()
            return None


def collate_fn(batch):
    # Filter out None elements in case of any faulty data points
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None

    try:
        # Handle 'mesh', 'cam_poses', and 'pos_enc3d' keys separately
        mesh_list = [item['mesh'] for item in batch]
        cam_poses_list = [item['cam_poses'] for item in batch]
        pos_enc3d_list = [item['pos_enc3d'] for item in batch]

        # Find the maximum length in the batch
        max_mesh_length = max(len(mesh) for mesh in mesh_list)
        max_cam_poses_length = max(len(cam) for cam in cam_poses_list)
        max_pos_enc3d_length = max(len(pos) for pos in pos_enc3d_list)

        # Pad the tensors to the maximum length
        def pad_tensor_list(tensor_list, max_length):
            padded_tensor_list = []
            for tensor in tensor_list:
                padded_tensor = torch.zeros((max_length, *tensor.shape[1:]), dtype=tensor.dtype)
                padded_tensor[:len(tensor)] = tensor
                padded_tensor_list.append(padded_tensor)
            return torch.stack(padded_tensor_list, dim=0)

        collated_meshes = pad_tensor_list(mesh_list, max_mesh_length)
        collated_cam_poses = pad_tensor_list(cam_poses_list, max_cam_poses_length)
        collated_pos_enc3d = pad_tensor_list(pos_enc3d_list, max_pos_enc3d_length)

        # Use default_collate for the rest of the keys, ensuring consistent shapes
        collated_batch = {}
        for key in batch[0]:
            if key in ['mesh', 'cam_poses', 'pos_enc3d']:
                continue
            if isinstance(batch[0][key], torch.Tensor):
                # If the tensors have different shapes, pad them as needed
                tensors = [d[key] for d in batch]
                max_shape = [max([tensor.size(dim) for tensor in tensors]) for dim in range(tensors[0].dim())]
                padded_tensors = []
                for tensor in tensors:
                    padding = [0] * (2 * tensor.dim())
                    for dim, size in enumerate(max_shape):
                        padding[2 * dim + 1] = size - tensor.size(dim)
                    padded_tensors.append(torch.nn.functional.pad(tensor, padding))
                collated_batch[key] = torch.stack(padded_tensors)
            else:
                collated_batch[key] = default_collate([d[key] for d in batch])

        # Add the padded tensors to the batch
        collated_batch['mesh'] = collated_meshes
        collated_batch['cam_poses'] = collated_cam_poses
        collated_batch['pos_enc3d'] = collated_pos_enc3d

        return collated_batch
    except Exception as e:
        print(f"Error during collation: {e}")
        raise

# def collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    # if shuffle:
    #     #print(dataset)
    #     return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_dataloaders(args, num_mesh_images = [5,5]):

    # if not os.path.isdir(args.data_path):
    #     args.data_path = '/raid/s2514643/brokenchairs/'
        
    good_folder_path = f'{args.data_path}/normals/'
    bad_folder_path = f'{args.data_path}/anomalies/'

    good_samples_list = glob.glob(good_folder_path+'/*/renders/*.png')
    good_samples_list = [str(path).replace(os.path.sep, '/') for path in good_samples_list]
    bad_samples_list = glob.glob(bad_folder_path+'/*/renders/*.png')
    bad_samples_list = [str(path).replace(os.path.sep, '/') for path in bad_samples_list]

    paths = [good_folder_path, bad_folder_path]
    samples = [good_samples_list, bad_samples_list]

    train_dataset = dataloader(args, paths, samples, num_mesh_images = num_mesh_images[0], mode = 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )          

    test_dataset = dataloader(args, paths, samples, num_mesh_images = num_mesh_images[1], mode = 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(test_dataset, shuffle=True, distributed=args.distributed),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )      

    return train_loader, test_loader 



