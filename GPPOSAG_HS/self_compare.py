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
from Algo.elo import *
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


model1_path = "./experiment/GPPO-SAG-HS_0.6/"
model2_path = "./experiment/GPPO-SAG-HS_0.7/"
# model_path_list = [
#     "./experiment/GPPO-SAG-HS_0.2/",
#     "./experiment/GPPO-SAG-HS_0.4/",
#     "./experiment/GPPO-SAG-HS_0.6/",
#     "./experiment/GPPO-SAG-HS_0.8/",
#     "./experiment/GPPO-SAG-HS_1.2/",
#     "./experiment/GPPO-SAG-HS_1.6/",
#     "./experiment/GPPO-SAG-HS_2.0/",
# ]

model_path_list = [
    "./experiment/PG-HS_1.0/",
    "./experiment/PPO-HS_0.3/",
    "./experiment/GPPO-SAG-HS_1.2/",
]

NUM_ROUNDS = 5001

# id_list = ["2000000.0","4000000.0","6000000.0","8000000.0","10000000.0",
#            "12000000.0","14000000.0","16000000.0","18000000.0","20000000.0",
#            "22000000.0","24000000.0","26000000.0","28000000.0","30000000.0",
#            "32000000.0","34000000.0","36000000.0","38000000.0","40000000.0",
#            "42000000.0","44000000.0","46000000.0","48000000.0","50000000.0",
#            "52000000.0","54000000.0","56000000.0","58000000.0","60000000.0",
#            "62000000.0","64000000.0","66000000.0","68000000.0","70000000.0",
#            "72000000.0","74000000.0","76000000.0","78000000.0","80000000.0",
#            "82000000.0","84000000.0","86000000.0","88000000.0","90000000.0",
#            "92000000.0","94000000.0","96000000.0","98000000.0","100000000.0",
#            "102000000.0","104000000.0","106000000.0","108000000.0","110000000.0",
#            "112000000.0","114000000.0","116000000.0","118000000.0","120000000.0",
#            "122000000.0","124000000.0","126000000.0","128000000.0","130000000.0",
#            "132000000.0","134000000.0","136000000.0","138000000.0","140000000.0",
#            "142000000.0","144000000.0","146000000.0","148000000.0","150000000.0",
#            "152000000.0","154000000.0","156000000.0","158000000.0","160000000.0",
#            "162000000.0","164000000.0","166000000.0","168000000.0","170000000.0",
#            "172000000.0","174000000.0","176000000.0","178000000.0","180000000.0",
#            "182000000.0","184000000.0","186000000.0","188000000.0","190000000.0",
#            "192000000.0","194000000.0","196000000.0","198000000.0","200000000.0",]

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
# id_list = id_list[0]
# model_type = ['GPPO','PG', 'PPO']
model_type = [
    'GPPO_0.2',
    'GPPO_0.4',
    'GPPO_1.2',
    ]

device_number = '7'

elo = EloImplementation()

model_list = []
for j in range(len(model_type)):
    temp_model = [Model(device=device_number, vs_stone_zero=False) for k in range(len(id_list))]
    
    for i in range(len(id_list)):
    # for i in range(87):
        checkpoint_path = model_path_list[j] + "Player1" + "_weights_" + id_list[i] + ".ckpt"
        checkpoint_states = torch.load(checkpoint_path)
        temp_model[i].get_model('Player1').load_state_dict(checkpoint_states)
        log.info("Loading model %s for %s from path: %s" % (model_type[j], id_list[i], checkpoint_path))
        elo.addPlayer(model_type[j]+id_list[i], rating=1200)
    
    model_list.append(temp_model)

game = Hearthstone(vs_StoneZero=False)
device = 'cuda:'+ device_number
env = Environment(game, device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
encoder = Encoder(model=auto_model, tokenizer=tokenizer)
encoder.to(device)

position, obs, options, done, episode_return = env.initial()
time_temp = time.time()
for i in range(NUM_ROUNDS):
    temp_agent_index1 = randint(0,len(model_list)-1)

    temp_checkpoint_index1 = randint(0,len(id_list)-1)


    temp_agent_index2 = randint(0,len(model_list)-1)

    temp_checkpoint_index2 = randint(0,len(id_list)-1)


    if i % 500 == 0:
        a = time.time()-time_temp
        print(f'cost time: {a} rounds: {i} || elo:')
        temp_elo = elo.getRatingList()
        print(temp_elo)
        time_temp = time.time()

        fileObject = open('elo.txt', 'w')  
        for ip in temp_elo:  
            fileObject.write(str(ip))  
            fileObject.write('\n')  
        fileObject.close()  
    while True:
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
                    agent_output, log_action_probs, _, _ = model_list[temp_agent_index1][temp_checkpoint_index1].select_forward('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
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
                    agent_output, log_action_probs, _, _ = model_list[temp_agent_index2][temp_checkpoint_index2].select_forward('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
                agent_options = action2option(agent_output, available_actions, options)
                if agent_options<0:
                    action_idx = torch.randint(len(options), (1, ))[0]
                else:
                    action_idx = agent_options

                action = options[action_idx]
        # log.info(action.FullPrint())
        position, obs, options, done, episode_return, _ = env.step(action)
        # log.info(env.game.game.FullPrint())
        if done:
            temp_Player1 = model_type[temp_agent_index1]+id_list[temp_checkpoint_index1]
            temp_Player2 = model_type[temp_agent_index2]+id_list[temp_checkpoint_index2]
            if episode_return > 0:  
                elo.recordMatch(temp_Player1, temp_Player2, winner=temp_Player1)
            elif episode_return < 0:
                elo.recordMatch(temp_Player1, temp_Player2, winner=temp_Player2)
            else:
                log.info("No winner???")
            position, obs, options, done, episode_return = env.initial()
            break


temp_elo = elo.getRatingList()
print(temp_elo)

fileObject = open('elo.txt', 'w')  
for ip in temp_elo:  
    fileObject.write(str(ip))  
    fileObject.write('\n')  
fileObject.close()  
