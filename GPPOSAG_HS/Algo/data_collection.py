
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

from Algo.Model.ModelWrapper import Model  
from Algo.utils import get_batch, log, create_buffers, create_optimizers, act 

mean_episode_return_buf = {
    p: deque(maxlen=100)
    for p in ['Player1', 'Player2']
}

def compute_loss(logits, targets):
    loss = ((logits.view(-1) - targets)**2).mean()
    return loss

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    card_id = batch['card_id'].to(device)
    minion_embed = batch['minion_embed'].to(device)
    weapon_embed = batch['weapon_embed'].to(device)
    secret_embed = batch['secret_embed'].to(device)
    hand = batch['hand'].to(device)
    minions = batch['minions'].to(device)
    heros = batch['heros'].to(device)
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(
        torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(card_id, minion_embed, secret_embed, weapon_embed, hand, minions, heros, num_options = None, actor = False)
        loss = compute_loss(learner_outputs, target)
        stats = {
            'mean_episode_return_' + position:
            torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]
                             ])).item(),
            'loss_' + position:
            loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

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

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' %
                           (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length  
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')
        ), 'The number of actor devices can not exceed the number of available devices'

    models = {}

    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)

    # Initialize queues
    '''
    mp.get_cocntext()：

    spawn:
    The parent process starts a fresh python interpreter process.
    Available on Unix and Windows. The default on Windows.

    fork:

    The parent process uses os.fork() to fork the Python interpreter. Available on Unix only. The default on Unix.

    forkserver

    When the program starts and selects the forkserver start method, a server process is started. From then on, whenever a new process is needed, the parent process connects to the server and requests that it fork a new process. The fork server process is single threaded so it is safe for it to use os.fork(). No unnecessary resources are inherited.

    Available on Unix platforms which support passing file descriptors over Unix pipes.
    '''

    ctx = mp.get_context('spawn')

    free_queue = {}
    full_queue = {}
    '''
    class multiprocessing.SimpleQueue:
    Support the following function:
    close(): 
    empty():
    get(): 
    put(item):
    '''
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
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)
    stat_keys = [
        'mean_episode_return_Player1',
        'loss_Player1',
        'mean_episode_return_Player2',
        'loss_Player2',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'Player1': 0, 'Player2': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device)
                          if flags.training_device != "cpu" else "cpu"))
        for k in ['Player1', 'Player2']:
            # learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"]["Player1"])
            optimizers[k].load_state_dict(
                checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(
                    learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

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
                           batch, optimizers['Player1'], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                #plogger.log(to_log)
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
            time.sleep(5)

            # if timer() - last_checkpoint_time > flags.save_interval * 60:
            #     checkpoint(frames)
            #     last_checkpoint_time = timer()
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
                'After %i (L:%i U:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f) Stats:\n%s',
                frames,
                position_frames['Player1'],
                position_frames['Player2'], fps, fps_avg,
                position_fps['Player1'], position_fps['Player2'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    #checkpoint(frames)
    #plogger.close()

