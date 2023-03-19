import argparse
import logging
import os
import random
import sys
import math
import time
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from Algo.Model.Cardsformer import Cardsformer
from Algo.sl_trainer import Trainer, TrainerConfig   #train

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)

    class MyDataset(Dataset):
        def __init__(self,data):
            self.data = data

        def __getitem__(self, index):
            data_step = self.data[index,:]
            x = torch.tensor(data_step[:-1,:], dtype=torch.float) 
            y = torch.tensor(data_step[1:,:], dtype=torch.float) 
            return x,y

        def __len__(self):
            return self.data.shape[0]

    class HSDataset(Dataset):

        def __init__(self, id_list):
            super(Dataset, self).__init__()
            self.data = {
                        'hand_card_embed': [],
                        'minion_embed': [],
                        'weapon_embed': [],
                        'secret_embed': [],
                        'hand_card_scalar': [],
                        'minion_scalar': [],
                        'hero_scalar': [],
                        'action_iter': [],
                        }
            
            self.id_list = id_list
            for id in id_list:
                data_path = '/data2/xingdp/yangyiming/Cardsformer-main/data/off_line_data' + str(id) + '.npy'
                cur_data = np.load(data_path, allow_pickle=True).item()
                print(f'load data in: {id}')
                for key in cur_data:
                    self.data[key].append(np.array(cur_data[key]))
            
            for key in cur_data:
                if key == 'action_iter':
                    type = torch.int
                else:
                    type = torch.float
                self.data[key] = torch.tensor(np.vstack(self.data[key]), dtype=type)
                print(f'{key}: {self.data[key].shape}')

        def __len__(self):
            return self.data['hand_card_embed'].shape[0]

        def __getitem__(self, index, id=False):

            action_iter = self.data['action_iter'][index]
            action_iter = torch.where(action_iter==-1, torch.zeros_like(action_iter), action_iter)
            action_iter_mask = torch.where(action_iter==-1, torch.zeros_like(action_iter, dtype=torch.bool), torch.ones_like(action_iter, dtype=torch.bool))
            if action_iter.dim() == 2:
                temp = action_iter.shape[0]
                action_info = {'action_type': action_iter[:,0],'target_card': action_iter[:,1],'target_entity': action_iter[:,2],'target_position': action_iter[:,3]}
                mask = {'action_type': torch.ones((temp,21), dtype=torch.bool),
                                    'target_card': torch.ones((temp,3), dtype=torch.bool),
                                    'target_entity': torch.ones((temp,17), dtype=torch.bool),
                                    'target_position': torch.ones((temp,7), dtype=torch.bool)}
                
                action_iter_mask = {'action_type': action_iter_mask[temp,0],
                                    'target_card': action_iter_mask[temp,1],
                                    'target_entity': action_iter_mask[temp,2],
                                    'target_position': action_iter_mask[temp,3]}
            else:
                action_info = {'action_type': action_iter[0],'target_card': action_iter[1],'target_entity': action_iter[2],'target_position': action_iter[3]}
                mask = {'action_type': torch.ones(21, dtype=torch.bool),
                                    'target_card': torch.ones(3, dtype=torch.bool),
                                    'target_entity': torch.ones(17, dtype=torch.bool),
                                    'target_position': torch.ones(7, dtype=torch.bool)}

                action_iter_mask = {'action_type': action_iter_mask[0],
                                    'target_card': action_iter_mask[1],
                                    'target_entity': action_iter_mask[2],
                                    'target_position': action_iter_mask[3]}
            
            return self.data['hand_card_embed'][index], \
                    self.data['minion_embed'][index], \
                    self.data['weapon_embed'][index], \
                    self.data['secret_embed'][index], \
                    self.data['hand_card_scalar'][index], \
                    self.data['minion_scalar'][index], \
                    self.data['hero_scalar'][index], \
                    mask, \
                    action_info,\
                    action_iter_mask

    class HSiterDataset(IterableDataset):

        def __init__(self, id_list):
            super(IterableDataset, self).__init__()
            self.data = {
                        'hand_card_embed': [],
                        'minion_embed': [],
                        'weapon_embed': [],
                        'secret_embed': [],
                        'hand_card_scalar': [],
                        'minion_scalar': [],
                        'hero_scalar': [],
                        'action_iter': [],
                        }
            self.id_list = id_list
                
            
        def __len__(self):
            pass
        def __iter__(self):
            
            file_list = self.id_list
            while True:
                for id in file_list:
                    time.sleep(0.1)
                    data_path = '/data2/xingdp/yangyiming/Cardsformer-main/data/off_line_data' + str(id) + '.npy'
                    cur_data = np.load(data_path, allow_pickle=True).item()
                    for key in cur_data:
                        self.data[key] = np.array(cur_data[key])
                    print(f'^^^^^^^^^^^^^^^^load:{id}^^^^^^^^^^^^^^^^^^^^^^')
                    index = 0
                    while(index < self.data['hand_card_embed'].shape[0]-1):
                        index+=1
                        action_iter = torch.tensor(self.data['action_iter'][index],dtype=torch.int)
                        action_iter = torch.where(action_iter==-1, torch.zeros_like(action_iter), action_iter)
                        action_info = {'action_type': action_iter[0],'target_card': action_iter[1],'target_entity': action_iter[2],'target_position': action_iter[3]}
                        mask = {'action_type': torch.ones(21, dtype=torch.bool),
                                    'target_card': torch.ones(3, dtype=torch.bool),
                                    'target_entity': torch.ones(17, dtype=torch.bool),
                                    'target_position': torch.ones(7, dtype=torch.bool)}
                        
                        
                        yield  torch.tensor(self.data['hand_card_embed'][index],dtype=torch.float), \
                                torch.tensor(self.data['minion_embed'][index],dtype=torch.float), \
                                torch.tensor(self.data['weapon_embed'][index],dtype=torch.float), \
                                torch.tensor(self.data['secret_embed'][index],dtype=torch.float), \
                                torch.tensor(self.data['hand_card_scalar'][index],dtype=torch.float), \
                                torch.tensor(self.data['minion_scalar'][index],dtype=torch.float), \
                                torch.tensor(self.data['hero_scalar'][index],dtype=torch.float), \
                                mask, \
                                action_info


    def main():

        train = [i for i in range(10)]
        test = [i for i in range(10, 12)]

        test_dataset = HSDataset(test)
        train_dataset = HSDataset(train)
        


        model = Cardsformer()

        tokens_per_epoch = 0
        train_epochs = 1000 # todo run a bigger model and longer, this is tiny

        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(max_epochs=train_epochs, batch_size=4096, learning_rate=1e-4,lr_decay=False, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch)
        trainer = Trainer(model, train_dataset, test_dataset, tconf)
        trainer.train()



    main()