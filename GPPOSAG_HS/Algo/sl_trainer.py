import math
import logging
import os
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from torch.nn import functional as F
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', 
#                         type=int, 
#                         help='node rank for distributed training')
# args = parser.parse_args()
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
# global_rank = dist.get_rank()

class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor, reduce=False) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduce:
            return loss.mean()
        else:
            return loss

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = 'sl_iter_HS.pkl'
    num_workers = 4 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model.cuda()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.test_loss1_store = np.zeros(config.max_epochs)
        self.test_loss2_store = np.zeros(config.max_epochs)
        self.loss_store = np.zeros(config.max_epochs)

        self.head_dim = {'action_type': 21,'target_card': 3,'target_entity': 17,'target_position': 7}
        
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    def save_checkpoint(self):
        ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(ckpt_model.state_dict(),self.config.ckpt_path)

    def train(self):
        model,config = self.model,self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        loss_CE = LabelSmoothingCrossEntropy()
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            train_dataset = self.train_dataset
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            # test_dataset = self.test_dataset
            # train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            #                         pin_memory=True, shuffle=(train_sampler is None),
            #                         sampler=train_sampler)  # 对于直接从硬盘读取数据num_workers=4 很重要 提高GPU利用率 linux
            # # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            # test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            #                        pin_memory=True, shuffle=True)

            test_dataset = self.test_dataset
            train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                    pin_memory=True, shuffle=True,)  # 对于直接从硬盘读取数据num_workers=4 很重要 提高GPU利用率 linux
            # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   pin_memory=True, shuffle=True)

            if is_train:
                loader = train_iter
            else:
                loader = test_iter
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, x in pbar:
                # place data on the correct device
                hand_card_embed = x[0].cuda()  
                minion_embed = x[1].cuda()  
                weapon_embed = x[2].cuda()  
                secret_embed = x[3].cuda()  
                hand_card_scalar = x[4].cuda()  
                minion_scalar = x[5].cuda()  
                hero_scalar = x[6].cuda()  
                for key in x[7].keys():
                    x[7][key] = x[7][key].cuda()
                    x[8][key] = x[8][key].cuda()
                    x[9][key] = x[9][key].cuda()
                mask = x[7]
                action_info = x[8]
                action_info_mask = x[9]

                # forward the model                    
                with torch.set_grad_enabled(is_train):
                    _, logit = model.sl_forward(hand_card_embed=hand_card_embed, 
                                                        minion_embed=minion_embed, 
                                                        secret_embed=secret_embed, 
                                                        weapon_embed=weapon_embed, 
                                                        hand_cards=hand_card_scalar, 
                                                        minions=minion_scalar, 
                                                        heros=hero_scalar, 
                                                        mask=mask, 
                                                        action_info=action_info)

                    loss = None
                    # for key in ['action_type', 'target_entity', 'target_position']:
                    for key in ['action_type']:
                        # temp_out = F.softmax(logit[key], dim=-1)
                        temp_loss = loss_CE(logit[key],action_info[key].long())
                        temp_loss = temp_loss.masked_fill(~action_info_mask[key], 0)
                        if loss is None:
                            loss = temp_loss
                        else:
                            loss = loss + temp_loss

                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += 1 # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")#. lr {lr:e}
                # time.sleep(0.01)
            if not is_train:
                loss = float(np.mean(losses))
                logger.info("sum loss: %f", loss)

                return loss
        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                loss = run_epoch('test')
                self.loss_store[epoch] = loss
                np.savetxt('sl_loss.txt',self.loss_store)
            # supports early stopping based on the test loss, or just save always if no test set is provided

            # if args.local_rank == 0:
            if True:
                if loss < best_loss :
                    best_loss=loss
                    print(loss)
                    self.save_checkpoint()
