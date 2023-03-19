
import os
import threading
import time
import timeit  
import pprint  
from collections import deque  
import numpy as np

import torch
from torch import multiprocessing as mp  
from torch import nn
from tensorboardX import SummaryWriter

from Algo.Model.ModelWrapper import Model 
from Algo.utils_self_play import get_batch, log, create_buffers, create_optimizers, act
from Algo.rl_loss import ReinforcementLoss
from distar.ctools.torch_utils.grad_clip import build_grad_clip

mean_winrate_buf = {
    p: deque(maxlen=10)
    for p in ['Player1', 'Player2']
}

def log_train(writter, train_infos, total_num_steps):
    """
    Log training info.
    :param train_infos: (dict) information about training update.
    :param total_num_steps: (int) total number of training env steps.
    """
    for k, v in train_infos.items():
        writter.add_scalars(k, {k: v}, total_num_steps)

def learn(position, actor_models, model, batch, optimizer, loss_func, flags, frames, lock, writter, grad_clip, use_factor=True):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')

    if True:
        card_id = batch['card_id'].to(device)
        minion_embed = batch['minion_embed'].to(device)
        weapon_embed = batch['weapon_embed'].to(device)
        secret_embed = batch['secret_embed'].to(device)
        hand = batch['hand'].to(device)
        minions = batch['minions'].to(device)
        heros = batch['heros'].to(device)
        reward = batch['reward'].to(device)
        done = batch['done'].to(device)
        
        mask = {}
        mask['action_type'] = batch['action_type_mask'].to(device)
        mask['target_card'] = batch['target_card_mask'].to(device)
        mask['target_entity'] = batch['target_entity_mask'].to(device)
        mask['target_position'] = batch['target_position_mask'].to(device)
        mask_head = {}
        mask_head['action_type'] = batch['action_type_mask_head'].to(device)
        mask_head['target_card'] = batch['target_card_mask_head'].to(device)
        mask_head['target_entity'] = batch['target_entity_mask_head'].to(device)
        mask_head['target_position'] = batch['target_position_mask_head'].to(device)
        action_info = {}
        action_info['action_type'] = batch['action_type'].to(device)
        action_info['target_card'] = batch['target_card'].to(device)
        action_info['target_entity'] = batch['target_entity'].to(device)
        action_info['target_position'] = batch['target_position'].to(device)
        behaviour_logp = {}
        behaviour_logp['action_type'] = batch['action_type_behavior_logprob'].to(device)
        behaviour_logp['target_card'] = batch['target_card_behavior_logprob'].to(device)
        behaviour_logp['target_entity'] = batch['target_entity_behavior_logprob'].to(device)
        behaviour_logp['target_position'] = batch['target_position_behavior_logprob'].to(device)
        teacher_logprob = {}
        teacher_logprob['action_type'] = batch['action_type_teacher_logprob'].to(device)
        teacher_logprob['target_card'] = batch['target_card_teacher_logprob'].to(device)
        teacher_logprob['target_entity'] = batch['target_entity_teacher_logprob'].to(device)
        teacher_logprob['target_position'] = batch['target_position_teacher_logprob'].to(device)

    with lock:
        # if frames < 1e7:
        #     loss_func.only_update_value = True
        #     model.only_update_baseline = True
        # else:
        #     model.only_update_baseline = False
        #     loss_func.only_update_value = False
        with torch.no_grad():
            model_output = model.train_forward(hand_card_embed=card_id, 
                                                    minion_embed=minion_embed, 
                                                    secret_embed=secret_embed, 
                                                    weapon_embed=weapon_embed, 
                                                    hand_cards=hand, 
                                                    minions=minions, 
                                                    heros=heros, 
                                                    behaviour_logp=behaviour_logp, 
                                                    teacher_logprob=teacher_logprob,
                                                    reward = reward,
                                                    mask=mask, 
                                                    mask_head=mask_head,
                                                    action_info=action_info,
                                                    done = done)
        T, B = model_output['reward'].size()
        factor = torch.ones((T,B), dtype=torch.float32, requires_grad=False).to(model_output['action_log_prob']['action_type'].device)
        factor_teacher = torch.ones_like(factor, requires_grad=False)
        factor_clip = torch.ones_like(factor, requires_grad=False)
        del T,B

        for head_type in ['action_type', 'target_entity', 'target_position']:
            model_output = model.train_forward(hand_card_embed=card_id, 
                                                    minion_embed=minion_embed, 
                                                    secret_embed=secret_embed, 
                                                    weapon_embed=weapon_embed, 
                                                    hand_cards=hand, 
                                                    minions=minions, 
                                                    heros=heros, 
                                                    behaviour_logp=behaviour_logp, 
                                                    teacher_logprob=teacher_logprob,
                                                    reward = reward,
                                                    mask=mask, 
                                                    mask_head=mask_head,
                                                    action_info=action_info,
                                                    done = done)
            temp_log_vars = loss_func.compute_loss(model_output, head_type, factor_clip, factor_teacher)
            loss = temp_log_vars['total_loss']
            if head_type == 'action_type':
                mean_winrate_buf[position].append(torch.nonzero(torch.sum(reward,dim=0)+1).shape[0]/reward.shape[1])
                winrate = torch.mean(
                        torch.stack([torch.tensor(_r) for _r in mean_winrate_buf[position]
                                    ])).item()

                writter.add_scalar('winrate', winrate, frames)
                stats = {
                    'winrate_' + position:
                    winrate,
                    'loss_' + position:
                    loss.item(),
                }
            
            log_train(writter, temp_log_vars, frames)
            optimizer.zero_grad()
            loss.backward()
            gradient = grad_clip.apply(model.parameters())
            log_train(writter, {'gradient/'+ head_type :gradient}, frames)
            optimizer.step()

            if use_factor:
                with torch.no_grad():
                    new_model_output = model.train_forward(hand_card_embed=card_id, 
                                                        minion_embed=minion_embed, 
                                                        secret_embed=secret_embed, 
                                                        weapon_embed=weapon_embed, 
                                                        hand_cards=hand, 
                                                        minions=minions, 
                                                        heros=heros, 
                                                        behaviour_logp=behaviour_logp, 
                                                        teacher_logprob=teacher_logprob,
                                                        reward = reward,
                                                        mask=mask, 
                                                        mask_head=mask_head,
                                                        action_info=action_info,
                                                        done = done)
                    factor_clip, factor, factor_teacher = loss_func.clip_factor(new_model_output, head_type, factor, factor_teacher)

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats


def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`"
            )

    checkpointpath = 'sl_iter_HS.pkl' 
    log_dir = flags.savedir + '/' + flags.xpid + '/' + 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writter = SummaryWriter(log_dir)

    T = flags.unroll_length  
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')
        ), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device,vs_stone_zero=False)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)

    # Initialize queues
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        _full_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device,vs_stone_zero=False)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)
    loss = ReinforcementLoss()
    grad_clip = build_grad_clip({'type': 'pytorch_norm', 'threshold': 1.0 })
    stat_keys = [
        'winrate_Player1',
        'loss_Player1',
        'winrate_Player2',
        'loss_Player2',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'Player1': 0, 'Player2': 0}

    # # Load models if any
    # if flags.load_model and os.path.exists(checkpointpath):
    #     print('load model')
    #     checkpoint_states = torch.load(checkpointpath, map_location=("cuda:" + str(flags.training_device)
    #                       if flags.training_device != "cpu" else "cpu"))
    #     learner_model.models['Player1'].load_state_dict(checkpoint_states['model_state_dict']['teacher'])
    #     for device in device_iterator:
    #         models[device].models['Player1'].load_state_dict(
    #             learner_model.models['Player1'].state_dict())

    # Starting actor processes
    actor_processes = []
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(target=act,
                                args=(i, device, free_queue[device],
                                      full_queue[device], models[device],
                                      buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i,
                        device,
                        position,
                        local_lock,
                        position_lock,
                        lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position],
                              full_queue[device][position],
                              buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position),
                           batch, optimizers['Player1'], loss, flags, frames, position_lock, writter, grad_clip)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['Player1'].put(m)
            free_queue[device]['Player2'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {
            'Player1': threading.Lock(),
            'Player2': threading.Lock()
        }
    position_locks = {
        'Player1': threading.Lock(),
        'Player2': threading.Lock()
    }

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['Player1', 'Player2']:
                thread = threading.Thread(target=batch_and_learn,
                                          name='batch-and-learn-%d' % i,
                                          args=(i, device, position,
                                                locks[device][position],
                                                position_locks['Player1']))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save(
            {
                'model_state_dict':
                {k: _models['Player1'].state_dict()
                for k in _models},
                'optimizer_state_dict':
                {k: optimizers[k].state_dict()
                 for k in optimizers},
                "stats": stats,
                'flags': vars(flags),
                'frames': frames,
                'position_frames': position_frames
            }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['Player1', 'Player1']:
            model_weights_dir = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' %
                                   (flags.savedir, flags.xpid, position +
                                    '_weights_' + str(frames) + '.ckpt')))
            torch.save(
                learner_model.get_model('Player1').state_dict(),
                model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        # last_checkpoint_time = timer() - flags.save_interval * 60
        last_save_frame = frames - (frames % flags.frame_interval)
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {
                k: position_frames[k]
                for k in position_frames
            }
            start_time = timer()
            time.sleep(60)

            if frames - last_save_frame > flags.frame_interval:
                checkpoint(frames - (frames % flags.frame_interval))
                last_save_frame = frames - (frames % flags.frame_interval)
            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {
                k: (position_frames[k] - position_start_frames[k]) /
                (end_time - start_time)
                for k in position_frames
            }
            log.info(
                'After %i (L:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f) Stats:\n%s',
                frames,
                position_frames['Player1'], fps, fps_avg,
                position_fps['Player1'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)


