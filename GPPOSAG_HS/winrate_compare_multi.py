from typing import OrderedDict
from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
from Env.utils import get_action_dict
from Algo.Model.ModelWrapper import Model  
from Algo.utils import action2option

from transformers import AutoModel, AutoTokenizer
import torch
from Algo.Model.Cardsformer import Encoder
import pandas as pd
from random import*
import logging
import time

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('Cardsformer')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

NUM_ROUNDS = 200
device_number = '7'

game = Hearthstone(vs_StoneZero=False)
device = 'cuda:'+ device_number
env = Environment(game, device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
encoder = Encoder(model=auto_model, tokenizer=tokenizer)
encoder.to(device)

def compare_vs_standard_model(num_rounds=NUM_ROUNDS,
                              model=None,
                              encoder=encoder,
                              env=env,
                              checkpoint_states_2=None):
    position, obs, options, done, episode_return = env.initial()
    time_temp = time.time()
    win = [0, 0]
    action_type_kl_np = np.zeros(num_rounds)
    target_entity_kl_np = np.zeros(num_rounds)
    target_position_kl_np = np.zeros(num_rounds)
    
    for i in range(num_rounds):
        temp_agent_index = randint(0,1)

        if i % 500 == 0:
            a = time.time()-time_temp
            # print(f'cost time: {a} rounds: {i}')
            time_temp = time.time()
        step=0
        action_type_kl = []
        target_entity_kl = []
        target_position_kl = []
        while True:
            step+=1
            num_options = len(options)
            # log.info(env.game.game.FullPrint())
            
            if position == "Player1":
                if num_options == 1:
                    action = options[0]
                else:
                    hand_card_embed = encoder.encode(obs['hand_card_names'], "hand_card")
                    minion_embed = encoder.encode(obs['minion_names'], "minion")
                    weapon_embed = encoder.encode(obs['weapon_names'], "weapon")
                    secret_embed = encoder.encode(obs['secret_names'], "secret")
                    available_actions, mask = get_action_dict(options,env.Hearthstone.game) # mask

                    with torch.no_grad():
                        agent_output, kl_dict = model[temp_agent_index].kl_estimate('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
                        action_type_kl.append(kl_dict['action_type'].cpu().item())
                        target_entity_kl.append(kl_dict['target_entity'].cpu().item())
                        target_position_kl.append(kl_dict['target_position'].cpu().item())
                    agent_options = action2option(agent_output, available_actions, options)
                    if agent_options<0:
                        action_idx = torch.randint(len(options), (1, ))[0]
                    else:
                        action_idx = agent_options
                    action = options[action_idx]


            elif position == "Player2":

                if num_options == 1:
                    action = options[0]
                else:

                    hand_card_embed = encoder.encode(obs['hand_card_names'], "hand_card")
                    minion_embed = encoder.encode(obs['minion_names'], "minion")
                    weapon_embed = encoder.encode(obs['weapon_names'], "weapon")
                    secret_embed = encoder.encode(obs['secret_names'], "secret")
                    available_actions, mask = get_action_dict(options,env.Hearthstone.game) # mask

                    with torch.no_grad():
                        agent_output, kl_dict = model[1-temp_agent_index].kl_estimate('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
                        action_type_kl.append(kl_dict['action_type'].cpu().item())
                        target_entity_kl.append(kl_dict['target_entity'].cpu().item())
                        target_position_kl.append(kl_dict['target_position'].cpu().item())
                    agent_options = action2option(agent_output, available_actions, options)
                    if agent_options<0:
                        action_idx = torch.randint(len(options), (1, ))[0]
                    else:
                        action_idx = agent_options

                    action = options[action_idx]
            # if num_options == 1:
            #     pass
            # else:
            #     print('------------------Available options--------------------')
            #     for i in range(len(options)):
            #         print(options[i].FullPrint() )
            #     print('------------------Actions choice--------------------')
            #     print(f'Actions in step {step}')
            #     print(action.FullPrint())
            #     print('------------------Current states--------------------')
            #     print(f'States after actions in step {step}')
            #     print(env.Hearthstone.game.FullPrint())
            #     print('------------------Agent action output--------------------')
            #     print(agent_output)
            #     print(available_actions)
            #     print('--------------------------Mask-----------------------------')
            #     print(mask)
            #     for k in mask_head.keys():
            #         print(f'{k}:{mask_head[k]}')
            #     print('-----------------------------------------------------------')

            position, obs, options, done, episode_return, _ = env.step(action)
            
            if done:
                action_type_kl_np[i] = np.mean(np.array(action_type_kl))
                target_entity_kl_np[i] = np.mean(np.array(target_entity_kl))
                target_position_kl_np[i] = np.mean(np.array(target_position_kl))
                if episode_return > 0:
                    if temp_agent_index == 0:
                        win[0]+=1
                    else:
                        win[1]+=1
                elif episode_return < 0:
                    if temp_agent_index == 0:
                        win[1]+=1
                    else:
                        win[0]+=1
                else:
                    log.info("No winner???")
                position, obs, options, done, episode_return = env.initial()
                break
        # print(step)
    # print(f'{model1_path} v.s. {checkpoint_states_2} : {win}')
    return win[1]/num_rounds, action_type_kl_np, target_entity_kl_np, target_position_kl_np

model1_path = "StoneZeroPretrainedModels/PPO_pretrain/Player1_weights_120000000.0.ckpt"
model2_path_1 = "experiment/compare/GPPO-SAG-HS_1.2/Player1_weights_"
model2_path_2 = "experiment/compare/PG-HS_1.0/Player1_weights_"
model2_path_3 = "experiment/compare/PPO-HS_0.3/Player1_weights_"

id_list = ["2000000.0","4000000.0","10000000.0",
           "14000000.0","20000000.0",
           "24000000.0","30000000.0",
           "34000000.0","40000000.0",
           "44000000.0","50000000.0",
           "54000000.0","60000000.0",
           "64000000.0","70000000.0",
           "74000000.0","80000000.0",
           "84000000.0","90000000.0",
           "94000000.0","100000000.0",
           "104000000.0","110000000.0",
           "114000000.0","120000000.0",
           "124000000.0","130000000.0",
           "134000000.0","140000000.0",
           "144000000.0","150000000.0",
           "154000000.0","160000000.0",
           "164000000.0","170000000.0",
           "174000000.0","178000000.0","180000000.0",
           "184000000.0","188000000.0","190000000.0",
           "194000000.0","198000000.0","200000000.0",]

model_1 = Model(device=device_number, vs_stone_zero=False)
model_2 = Model(device=device_number, vs_stone_zero=False)

checkpoint_states_1 = torch.load(model1_path)
model_1.get_model('Player1').load_state_dict(checkpoint_states_1)

import numpy as np
winrate_np = np.zeros(len(id_list))
for j in range(3):
    for model2_path in [model2_path_1, model2_path_2, model2_path_3]:
        action_type_kl_np = np.zeros([len(id_list),NUM_ROUNDS])
        target_entity_kl_np = np.zeros([len(id_list),NUM_ROUNDS])
        target_position_kl_np = np.zeros([len(id_list),NUM_ROUNDS])
        for i in range(len(id_list)):
            checkpoint_path = model2_path + id_list[i] +'.ckpt'
            checkpoint_states_2 = torch.load(checkpoint_path)
            model_2.get_model('Player1').load_state_dict(checkpoint_states_2)
            winrate_np[i], action_type_kl_np[i], target_entity_kl_np[i], target_position_kl_np[i] = compare_vs_standard_model(model=[model_1, model_2], checkpoint_states_2=checkpoint_path)

        print(model2_path)
        print(id_list)
        print(winrate_np.tolist())
        if model2_path == model2_path_1:
            temp = 'GPPO_SAG'
        elif model2_path == model2_path_2:
            temp = 'PG'
        else:
            temp = 'PPO'
        np.save(f'seed_{j}_{temp}_action_type_kl.npy',action_type_kl_np)
        np.save(f'seed_{j}_{temp}_target_entity_kl.npy',target_entity_kl_np)
        np.save(f'seed_{j}_{temp}_target_position_kl.npy',target_position_kl_np)
