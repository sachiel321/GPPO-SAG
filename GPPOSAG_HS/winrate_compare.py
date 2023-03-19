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

model1_path = "model_path_1"

model2_path = "model_path_2"


NUM_ROUNDS = 200
device_number = '0'

model_1 = Model(device=device_number, vs_stone_zero=False)
model_2 = Model(device=device_number, vs_stone_zero=False)

model = [model_1, model_2]

checkpoint_states_1 = torch.load(model1_path)
model_1.get_model('Player1').load_state_dict(checkpoint_states_1)

checkpoint_path = model2_path
checkpoint_states_2 = torch.load(model2_path)
model_2.get_model('Player1').load_state_dict(checkpoint_states_2)

del checkpoint_states_1, checkpoint_states_2

game = Hearthstone(vs_StoneZero=False)
device = 'cuda:'+ device_number
env = Environment(game, device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
encoder = Encoder(model=auto_model, tokenizer=tokenizer)
encoder.to(device)

position, obs, options, done, episode_return = env.initial()
time_temp = time.time()
win = [0, 0]
for i in range(NUM_ROUNDS):
    temp_agent_index = randint(0,1)

    if i % 500 == 0:
        a = time.time()-time_temp
        print(f'cost time: {a} rounds: {i}')
        time_temp = time.time()
    step=0
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
                    agent_output, log_action_probs, mask, mask_head = model[temp_agent_index].select_forward('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
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
                    agent_output, log_action_probs, mask, mask_head = model[1-temp_agent_index].select_forward('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
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
print(f'{model1_path} v.s. {model2_path} : {win}')

