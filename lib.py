import os

import torch
import cv2
import numpy as np
from natsort import natsorted
import glob


class Pre_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 video_folder,
                 extension='jpg',
                 transforms=None):

        video_dir = natsorted(os.listdir(video_folder))
        self.videos = []
        self.futures = []
        self.T = opt.T
        self.transforms = transforms
        for i in range(len(video_dir)):
            frame_list = natsorted(glob.glob(video_folder + video_dir[i] + '/*.' + extension))
            # print(frame_list)

            for j in range(len(frame_list) - opt.T):
                # print(frames[0])
                # print(frame_list[j + opt.T+1])
                video = [frame_list[j:j + opt.T][k] for k in range(opt.T)]
                future = [frame_list[j + opt.T]]

                # print(video)ramerame
                self.videos.append(video)
                self.futures.append(future)

        # print(len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_list = self.videos[idx]
        future_list = self.futures[idx]
        # print(future_list)

        # video = np.array([self.transforms(cv2.imread(video_list[i])).numpy() for i in range(len(video_list))])
        video = np.array([self.transforms(cv2.cvtColor(cv2.imread(video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in
                          range(len(video_list))])
        future = self.transforms(cv2.cvtColor(cv2.imread(future_list[0]), cv2.COLOR_BGR2RGB)).numpy()
        # for i in range(len(video_list)):
        #     self.transforms(cv2.imread(video_list[i])).cat([video],dim=0)
        #
        # for j in range(len(future_video_list)):
        #     self.transforms(cv2.imread(future_video_list[j])).append(future_video)

        return torch.from_numpy(video), torch.from_numpy(future)


class VideoDataloader(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 video_folder,
                 transforms=None):


        video_dir = natsorted(os.listdir(video_folder))
        self.videos = []
        self.T = opt.T
        self.transforms = transforms
        for i in range(len(video_dir)):
            frame_list = natsorted(glob.glob(video_folder + video_dir[i] + '/*.jpg'))

            for j in range(len(frame_list)- opt.T*2 + 1):

                #print(frames[0])
                video = [frame_list[j:j+opt.T*2][k] for k in range (opt.T*2)]
                #print(video)
                self.videos.append(video)
        #print(len(self.videos))



    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_list = self.videos[idx][:self.T]
        future_video_list = self.videos[idx][self.T:]

        video = np.array([self.transforms(cv2.cvtColor(cv2.imread(video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in range(len(video_list))])
        future_video = np.array([self.transforms(cv2.cvtColor(cv2.imread(future_video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in range(len(future_video_list))])

        # for i in range(len(video_list)):
        #     self.transforms(cv2.imread(video_list[i])).cat([video],dim=0)
        #
        # for j in range(len(future_video_list)):
        #     self.transforms(cv2.imread(future_video_list[j])).append(future_video)

        return torch.from_numpy(video),torch.from_numpy(future_video)