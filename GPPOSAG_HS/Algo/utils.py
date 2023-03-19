import torch
import typing
import traceback
import numpy as np
import random
from torch import multiprocessing as mp
import torch.nn.functional as F
import time
from copy import deepcopy

from Env.EnvWrapper import Environment
from Env.Hearthstone import Hearthstone, log
from transformers import AutoModel, AutoTokenizer
from Algo.Model.Cardsformer import Encoder
from Env.utils import get_action_dict, get_action_mask
from StoneZeroModel.PredictionModel import PredictionModel

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

head_name = ['action_type', 'target_card', 'target_entity', 'target_position']

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    
    batch = {}
    for key in buffers:
        batch[key] = torch.stack([buffers[key][m] for m in indices], dim=1)

    for m in indices:
        free_queue.put(m)

    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['Player1']
    optimizers = {}
    # modified for sigle model
    for position in positions:
        optimizer = torch.optim.Adam(
            learner_model.parameters(position),
            lr=flags.learning_rate, )
        optimizers[position] = optimizer
    return optimizers

def action2option(agent_output, available_actions, options):
    action_type = agent_output['action_type'].cpu().item()
    action_type = int(action_type)
    if action_type == 0:
        action_idx = available_actions['EndTurnTask']
        return action_idx
    elif action_type < 12:
        if action_type-1 not in available_actions['PlayCardTask'].keys():
            return -1
        else:
            dict_i = available_actions['PlayCardTask'][action_type-1]
            target_entity = agent_output['target_entity'].cpu().item()
            target_entity = int(target_entity)
            if target_entity not in dict_i.keys():
                return -2
            else:
                target_position = agent_output['target_position'].cpu().item()
                target_position = int(target_position)
                if isinstance(dict_i[target_entity], dict):
                    dict_j = dict_i[target_entity]
                    if target_position not in dict_j.keys():
                        return random.sample(list(dict_j.values()),1)
                    else:
                        action_idx = dict_j[target_position]
                        return action_idx
                else:
                    action_idx = dict_i[target_entity]
                    return action_idx
    elif action_type >= 12 and action_type <= 18:
        if action_type-12 not in available_actions['MinionAttackTask']:
            return -1
        else:
            dict_i = available_actions['MinionAttackTask'][action_type-12]
            if isinstance(dict_i, dict):
                target_entity = agent_output['target_entity'].cpu().item()
                target_entity = int(target_entity)
                if target_entity not in dict_i.keys():
                    return random.sample(list(dict_i.values()),1)
                else:
                    action_idx = dict_i[target_entity]
                    return action_idx
    elif action_type == 19:
        target_entity = agent_output['target_entity'].cpu().item()
        target_entity = int(target_entity)
        if target_entity not in available_actions['HeroAttackTask'].keys():
            return random.sample(list(available_actions['HeroAttackTask'].values()),1)
        else:
            action_idx = available_actions['HeroAttackTask'][target_entity]
            return action_idx
    elif action_type == 20:
        target_card = agent_output['target_card'].cpu().item()
        target_card = int(target_card)
        action_idx = available_actions['Discovery'][target_card]
        return action_idx

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length  
    positions = ['Player1']
    
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool), 
                episode_return=dict(size=(T,), dtype=torch.float32),
                reward=dict(size=(T,), dtype=torch.float32),
                # TODO
                card_id=dict(size=(T, 11, 768), dtype=torch.float32), 
                minion_embed=dict(size=(T, 14, 768), dtype=torch.float32),
                weapon_embed=dict(size=(T, 2, 768), dtype=torch.float32),
                secret_embed=dict(size=(T, 5, 768), dtype=torch.float32),
                hand=dict(size=(T, 11, 20), dtype=torch.float32),
                minions=dict(size=(T, 14, 23), dtype=torch.float32),
                heros=dict(size=(T, 2, 29), dtype=torch.float32),

                action_type_mask=dict(size=(T,), dtype=torch.bool),
                target_card_mask=dict(size=(T,), dtype=torch.bool),
                target_entity_mask=dict(size=(T,), dtype=torch.bool),
                target_position_mask=dict(size=(T,), dtype=torch.bool),
                
                action_type_mask_head=dict(size=(T, 21), dtype=torch.bool),
                target_card_mask_head=dict(size=(T, 3), dtype=torch.bool),
                target_entity_mask_head=dict(size=(T, 17), dtype=torch.bool),
                target_position_mask_head=dict(size=(T, 7), dtype=torch.bool),
                
                action_type=dict(size=(T, 1), dtype=torch.int32),
                target_card=dict(size=(T, 1), dtype=torch.int32),
                target_entity=dict(size=(T, 1), dtype=torch.int32),
                target_position=dict(size=(T, 1), dtype=torch.int32),

                action_type_teacher_logprob=dict(size=(T, 21), dtype=torch.float32),
                target_card_teacher_logprob=dict(size=(T, 3), dtype=torch.float32),
                target_entity_teacher_logprob=dict(size=(T, 17), dtype=torch.float32),
                target_position_teacher_logprob=dict(size=(T, 7), dtype=torch.float32),

                action_type_behavior_logprob=dict(size=(T, 21), dtype=torch.float32),
                target_card_behavior_logprob=dict(size=(T, 3), dtype=torch.float32),
                target_entity_behavior_logprob=dict(size=(T, 17), dtype=torch.float32),
                target_position_behavior_logprob=dict(size=(T, 7), dtype=torch.float32),
            )

            _buffers: Buffers = {key: [] for key in specs}

            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        if key == 'done':
                            _buffer = torch.ones(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                        else:
                            _buffer = torch.zeros(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        if key ==  'done':
                            _buffer = torch.ones(**specs[key]).to(torch.device('cpu')).share_memory_()
                        else:
                            _buffer = torch.zeros(**specs[key]).to(torch.device('cpu')).share_memory_()

                    _buffers[key].append(_buffer)
                
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    #TODO: add reward behaviour_logp teacher_logit
    positions = ['Player1']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)
        
        env = Hearthstone()
        env = Environment(env, device)

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        encoder = Encoder(model=auto_model, tokenizer=tokenizer)
        encoder.to(device)
        position, obs, options, done, episode_return = env.initial()
        
        prediction_model = PredictionModel()
        checkpoint_states = torch.load("./StoneZeroPretrainedModels/prediction_model.tar", map_location='cpu')['model_state_dict']
        new_state_dict = typing.OrderedDict()
        for k, v in checkpoint_states.items():
            name = k[7:]
            new_state_dict[name] = v
        
        prediction_model.load_state_dict(new_state_dict)
        prediction_model.to(device)
        prediction_model.eval()
        
        while True:
            if free_queue['Player1'].empty() or free_queue['Player1'].empty():
                time.sleep(2)
            else:
                index_1 = free_queue['Player1'].get()
                index = {'Player1':index_1}
                t = {'Player1':0}
                flag_full = {'Player1':False}
                while True:
                    num_options = len(options)
                    p = position
                    if num_options == 1:
                        action = options[0]
                    else:
                        hand_card_embed = encoder.encode(obs['hand_card_names'], "hand_card")
                        minion_embed = encoder.encode(obs['minion_names'], "minion")
                        weapon_embed = encoder.encode(obs['weapon_names'], "weapon")
                        secret_embed = encoder.encode(obs['secret_names'], "secret")
                        available_actions, mask = get_action_dict(options,env.Hearthstone.game) # mask

                        if position == 'Player1':
                            with torch.no_grad():
                                _, teacher_logprob, mask_teacher, mask_head_teacher = model.select_forward('teacher',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
                                agent_output, log_action_probs, mask, mask_head = model.select_forward('actor',hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor=True)
                            agent_options = action2option(agent_output, available_actions, options)
                            if agent_options<0:
                                _action_idx = torch.randint(len(options), (1, ))[0]
                            else:
                                _action_idx = agent_options
                            action = options[_action_idx]
                            
                            if flag_full[p] == False:
                                buffers[p]['card_id'][index[p]][t[p], ...] = hand_card_embed
                                buffers[p]['minion_embed'][index[p]][t[p], ...] = minion_embed
                                buffers[p]['weapon_embed'][index[p]][t[p], ...] = weapon_embed
                                buffers[p]['secret_embed'][index[p]][t[p], ...] = secret_embed
                                buffers[p]['hand'][index[p]][t[p], ...] =	obs["hand_card_scalar"]
                                buffers[p]['minions'][index[p]][t[p], ...] = obs["minion_scalar"]
                                buffers[p]['heros'][index[p]][t[p], ...] = obs["hero_scalar"]

                                buffers[p]['action_type_mask'][index[p]][t[p], ...] = mask['action_type']
                                buffers[p]['target_card_mask'][index[p]][t[p], ...] = mask['target_card']
                                buffers[p]['target_entity_mask'][index[p]][t[p], ...] = mask['target_entity']
                                buffers[p]['target_position_mask'][index[p]][t[p], ...] = mask['target_position']
                                
                                buffers[p]['action_type_mask_head'][index[p]][t[p], ...] = mask_head['action_type']
                                buffers[p]['target_card_mask_head'][index[p]][t[p], ...] = mask_head['target_card']
                                buffers[p]['target_entity_mask_head'][index[p]][t[p], ...] = mask_head['target_entity']
                                buffers[p]['target_position_mask_head'][index[p]][t[p], ...] = mask_head['target_position']

                                buffers[p]['action_type'][index[p]][t[p], ...] = agent_output['action_type']
                                buffers[p]['target_card'][index[p]][t[p], ...] = agent_output['target_card']
                                buffers[p]['target_entity'][index[p]][t[p], ...] = agent_output['target_entity']
                                buffers[p]['target_position'][index[p]][t[p], ...] = agent_output['target_position']

                                buffers[p]['action_type_teacher_logprob'][index[p]][t[p], ...] = teacher_logprob['action_type']
                                buffers[p]['target_card_teacher_logprob'][index[p]][t[p], ...] = teacher_logprob['target_card']
                                buffers[p]['target_entity_teacher_logprob'][index[p]][t[p], ...] = teacher_logprob['target_entity']
                                buffers[p]['target_position_teacher_logprob'][index[p]][t[p], ...] = teacher_logprob['target_position']

                                buffers[p]['action_type_behavior_logprob'][index[p]][t[p], ...] = log_action_probs['action_type']
                                buffers[p]['target_card_behavior_logprob'][index[p]][t[p], ...] = log_action_probs['target_card']
                                buffers[p]['target_entity_behavior_logprob'][index[p]][t[p], ...] = log_action_probs['target_entity']
                                buffers[p]['target_position_behavior_logprob'][index[p]][t[p], ...] = log_action_probs['target_position']
                                
                                buffers[p]['done'][index[p]][t[p], ...] = done
                            else:
                                buffers[p]['card_id'][index[p]][:-1] = buffers[p]['card_id'][index[p]][1:].clone()
                                buffers[p]['minion_embed'][index[p]][:-1] = buffers[p]['minion_embed'][index[p]][1:].clone()
                                buffers[p]['weapon_embed'][index[p]][:-1] = buffers[p]['weapon_embed'][index[p]][1:].clone()
                                buffers[p]['secret_embed'][index[p]][:-1] = buffers[p]['secret_embed'][index[p]][1:].clone()
                                buffers[p]['hand'][index[p]][:-1] =	buffers[p]['hand'][index[p]][1:].clone()
                                buffers[p]['minions'][index[p]][:-1] = buffers[p]['minions'][index[p]][1:].clone()
                                buffers[p]['heros'][index[p]][:-1] = buffers[p]['heros'][index[p]][1:].clone()

                                buffers[p]['action_type_mask'][index[p]][:-1] = buffers[p]['action_type_mask'][index[p]][1:].clone()
                                buffers[p]['target_card_mask'][index[p]][:-1] = buffers[p]['target_card_mask'][index[p]][1:].clone()
                                buffers[p]['target_entity_mask'][index[p]][:-1] = buffers[p]['target_entity_mask'][index[p]][1:].clone()
                                buffers[p]['target_position_mask'][index[p]][:-1] = buffers[p]['target_position_mask'][index[p]][1:].clone()
                                
                                buffers[p]['action_type_mask_head'][index[p]][:-1] = buffers[p]['action_type_mask_head'][index[p]][1:].clone()
                                buffers[p]['target_card_mask_head'][index[p]][:-1] = buffers[p]['target_card_mask_head'][index[p]][1:].clone()
                                buffers[p]['target_entity_mask_head'][index[p]][:-1] = buffers[p]['target_entity_mask_head'][index[p]][1:].clone()
                                buffers[p]['target_position_mask_head'][index[p]][:-1] = buffers[p]['target_position_mask_head'][index[p]][1:].clone()

                                buffers[p]['action_type'][index[p]][:-1] = buffers[p]['action_type'][index[p]][1:].clone()
                                buffers[p]['target_card'][index[p]][:-1] = buffers[p]['target_card'][index[p]][1:].clone()
                                buffers[p]['target_entity'][index[p]][:-1] = buffers[p]['target_entity'][index[p]][1:].clone()
                                buffers[p]['target_position'][index[p]][:-1] = buffers[p]['target_position'][index[p]][1:].clone()

                                buffers[p]['action_type_teacher_logprob'][index[p]][:-1] = buffers[p]['action_type_teacher_logprob'][index[p]][1:].clone()
                                buffers[p]['target_card_teacher_logprob'][index[p]][:-1] = buffers[p]['target_card_teacher_logprob'][index[p]][1:].clone()
                                buffers[p]['target_entity_teacher_logprob'][index[p]][:-1] = buffers[p]['target_entity_teacher_logprob'][index[p]][1:].clone()
                                buffers[p]['target_position_teacher_logprob'][index[p]][:-1] = buffers[p]['target_position_teacher_logprob'][index[p]][1:].clone()

                                buffers[p]['action_type_behavior_logprob'][index[p]][:-1] = buffers[p]['action_type_behavior_logprob'][index[p]][1:].clone()
                                buffers[p]['target_card_behavior_logprob'][index[p]][:-1] = buffers[p]['target_card_behavior_logprob'][index[p]][1:] .clone()
                                buffers[p]['target_entity_behavior_logprob'][index[p]][:-1] = buffers[p]['target_entity_behavior_logprob'][index[p]][1:].clone()
                                buffers[p]['target_position_behavior_logprob'][index[p]][:-1] = buffers[p]['target_position_behavior_logprob'][index[p]][1:].clone()
                                buffers[p]['done'][index[p]][:-1] = buffers[p]['done'][index[p]][1:].clone()

                                buffers[p]['done'][index[p]][-1] = done
                                buffers[p]['card_id'][index[p]][-1] = hand_card_embed
                                buffers[p]['minion_embed'][index[p]][-1] = minion_embed
                                buffers[p]['weapon_embed'][index[p]][-1] = weapon_embed
                                buffers[p]['secret_embed'][index[p]][-1] = secret_embed
                                buffers[p]['hand'][index[p]][-1] =	obs["hand_card_scalar"]
                                buffers[p]['minions'][index[p]][-1] = obs["minion_scalar"]
                                buffers[p]['heros'][index[p]][-1] = obs["hero_scalar"]

                                buffers[p]['action_type_mask'][index[p]][-1] = mask['action_type']
                                buffers[p]['target_card_mask'][index[p]][-1] = mask['target_card']
                                buffers[p]['target_entity_mask'][index[p]][-1] = mask['target_entity']
                                buffers[p]['target_position_mask'][index[p]][-1] = mask['target_position']
                                
                                buffers[p]['action_type_mask_head'][index[p]][-1] = mask_head['action_type']
                                buffers[p]['target_card_mask_head'][index[p]][-1] = mask_head['target_card']
                                buffers[p]['target_entity_mask_head'][index[p]][-1] = mask_head['target_entity']
                                buffers[p]['target_position_mask_head'][index[p]][-1] = mask_head['target_position']

                                buffers[p]['action_type'][index[p]][-1] = agent_output['action_type']
                                buffers[p]['target_card'][index[p]][-1] = agent_output['target_card']
                                buffers[p]['target_entity'][index[p]][-1] = agent_output['target_entity']
                                buffers[p]['target_position'][index[p]][-1] = agent_output['target_position']

                                buffers[p]['action_type_teacher_logprob'][index[p]][-1] = teacher_logprob['action_type']
                                buffers[p]['target_card_teacher_logprob'][index[p]][-1] = teacher_logprob['target_card']
                                buffers[p]['target_entity_teacher_logprob'][index[p]][-1] = teacher_logprob['target_entity']
                                buffers[p]['target_position_teacher_logprob'][index[p]][-1] = teacher_logprob['target_position']

                                buffers[p]['action_type_behavior_logprob'][index[p]][-1] = log_action_probs['action_type']
                                buffers[p]['target_card_behavior_logprob'][index[p]][-1] = log_action_probs['target_card']
                                buffers[p]['target_entity_behavior_logprob'][index[p]][-1] = log_action_probs['target_entity']
                                buffers[p]['target_position_behavior_logprob'][index[p]][-1] = log_action_probs['target_position']
                        else:
                            hand_card_embed = encoder.encode(obs['hand_card_names'])
                            minion_embed = encoder.encode(obs['minion_names'])
                            weapon_embed = encoder.encode(obs['weapon_names'])
                            secret_embed = encoder.encode(obs['secret_names'])
                            with torch.no_grad():
                                next_state = prediction_model([hand_card_embed, minion_embed, weapon_embed, obs['hand_card_scalar_batch'], obs['minion_scalar_batch'], obs['hero_scalar_batch']])
                            obs['next_minion_scalar'] = next_state[0]
                            obs['next_hero_scalar'] = next_state[1]
                            with torch.no_grad():
                                agent_output = model.stone_zero_select_options(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor = True)
                            agent_output = agent_output.argmax()
                            _action_idx = int(agent_output.cpu().detach().numpy())
                            action = options[_action_idx]
                                
                        
                            
                    next_position, next_obs, options, done, episode_return, _ = env.step(action)
                    reward = 0
                    if position == 'Player1':
                    
                        if flag_full[position] == False:
                            buffers[position]['reward'][index[position]][t[position], ...] = reward
                            
                        else:
                            buffers[position]['reward'][index[position]][:-1] =buffers[position]['reward'][index[position]][1:]
                            buffers[position]['reward'][index[position]][-1] = reward

                    position = next_position
                    obs = next_obs
                    
                    if done:
                        for pp in positions:

                            episode_return = episode_return if pp == 'Player1' else -episode_return
                            if torch.is_tensor(episode_return):
                                episode_return = episode_return.cpu().item()
                        
                            if flag_full[pp] == False:
                                buffers[pp]['reward'][index[pp]][t[pp], ...] = episode_return
                                buffers[pp]['done'][index[pp]][t[pp], ...] = done
                                buffers[pp]['reward'][index[pp]][t[pp]+1:] = 0
                                buffers[pp]['done'][index[pp]][t[pp]+1:] = True
                                buffers[pp]['card_id'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['minion_embed'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['weapon_embed'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['secret_embed'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['hand'][index[pp]][t[pp]+1:]  =	0
                                buffers[pp]['minions'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['heros'][index[pp]][t[pp]+1:]  = 0

                                buffers[pp]['action_type_mask'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_card_mask'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_entity_mask'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_position_mask'][index[pp]][t[pp]+1:]  = 0
                                
                                buffers[pp]['action_type_mask_head'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_card_mask_head'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_entity_mask_head'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_position_mask_head'][index[pp]][t[pp]+1:]  = 0
                                
                                
                                buffers[pp]['action_type'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_card'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_entity'][index[pp]][t[pp]+1:]  = 0
                                buffers[pp]['target_position'][index[pp]][t[pp]+1:]  = 0

                                buffers[pp]['action_type_teacher_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_card_teacher_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_entity_teacher_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_position_teacher_logprob'][index[pp]][t[pp]+1:]  = -1e9

                                buffers[pp]['action_type_behavior_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_card_behavior_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_entity_behavior_logprob'][index[pp]][t[pp]+1:]  = -1e9
                                buffers[pp]['target_position_behavior_logprob'][index[pp]][t[pp]+1:]  = -1e9
                            else:
                                buffers[pp]['reward'][index[pp]][:-1] = buffers[pp]['reward'][index[pp]][index[pp]][1:]
                                buffers[pp]['reward'][index[pp]][-1] = episode_return
                                # buffers[pp]['reward'][index[pp]][:-1] += episode_return/2
                                
                                buffers[pp]['done'][index[pp]][:-1] = buffers[pp]['done'][index[pp]][index[pp]][1:]
                                buffers[pp]['done'][index[pp]][-1] = done
                        position, obs, options, done, episode_return = env.initial()          
                        break
                    if p == 'Player1' and num_options != 1:
                        t[p] += 1
                        if t[p] > T-1:
                            flag_full[p] = True

                for p in positions:
                    full_queue[p].put(index[p])

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

