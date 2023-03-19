import os
import threading
import time
import timeit  
import numpy as np

import torch
from torch import multiprocessing as mp  

from Algo.Model.ModelWrapper import Model  
from Algo.utils_self_play import log, elo_act
from Algo.elo import *

def create_buffers(device_iterator):
    
    buffers = {}
    for device in device_iterator:
        
        buffers[device] = {}
        _buffers = {'Player1_id':-1, 'Player2_id':-1,'win_side_id':-1}
        for _ in range(50):
            for k in _buffers.keys():
                _buffers[k].append([])    
        buffers[device] = _buffers
    return buffers

def get_data(free_queue,
              full_queue,
              buffers,
              lock):
    with lock:
        indices = full_queue.get()
    
    batch = {}
    for key in buffers:
        batch[key] = buffers[key][indices]

    free_queue.put(indices)

    return batch

def elo_stat(elo, player_list, batch, lock, ):

    if True:
        player1_id = batch['Player1_id']
        player2_id = batch['Player2_id']
        win_side_id = batch['win_side_id']
        
    with lock:
        elo.recordMatch(player1_id, player2_id, winner=win_side_id)
        return 0

def eval(flags):

    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`"
            )

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')
        ), 'The number of actor devices can not exceed the number of available devices'

    model_path = "./experiment/rl_model_path/"
    total_frames = 6000
    id_list = ["2000000.0","4000000.0","6000000.0","8000000.0","10000000.0",
            "20000000.0","30000000.0","40000000.0","50000000.0","60000000.0",
            "70000000.0","80000000.0","90000000.0","100000000.0","110000000.0","120000000.0",
            "130000000.0","140000000.0","150000000.0","160000000.0","170000000.0",] # 11个
    player_dict = {}
    for i in range(len(id_list)):
        player_dict[i] = id_list[i]
    model_type = ['PPO']

    elo = EloImplementation()
    for i in range(len(id_list)):
        elo.addPlayer(model_type[0]+id_list[i], rating=1200)
    
    models = {}
    
    for device in device_iterator:
        model = [Model(device=device, vs_stone_zero=False) for i in range(len(id_list))]
        for i in range(len(id_list)):
            checkpoint_path = model_path + "Player1" + "_weights_" + id_list[i] + ".ckpt"
            checkpoint_states = torch.load(checkpoint_path)
            model[i].get_model('Player1').load_state_dict(checkpoint_states)
            model[i].share_memory()
            model[i].eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(device_iterator)

    # Initialize queues
    ctx = mp.get_context('spawn')

    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = ctx.SimpleQueue(),
        _full_queue = ctx.SimpleQueue(),
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue


    frames = 0
    # Starting actor processes
    actor_processes = []
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(target=elo_act,
                                args=(i, device, free_queue[device],
                                      full_queue[device], models[device],
                                      buffers[device], player_dict))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i,
                        device,
                        local_lock,
                        position_lock,
                        lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames
        while frames < flags.total_frames:
            batch = get_data(free_queue[device],
                              full_queue[device],
                              buffers[device], flags, local_lock)
            _stats = elo_stat(elo, player_dict, batch, position_lock,)

            with lock:
                frames += 1

    for device in device_iterator:
        for m in range(50):
            free_queue[device].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = threading.Lock()
    position_locks = threading.Lock()

    for device in device_iterator:
        for i in range(flags.num_threads):
            thread = threading.Thread(target=batch_and_learn,
                                        name='batch-and-learn-%d' % i,
                                        args=(i, device,
                                            locks[device],
                                            position_locks))
            thread.start()
            threads.append(thread)

    fps_log = []
    timer = timeit.default_timer
    try:
        while frames < total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(60)

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            log.info(
                'After %i frames: @ %.1f fps (avg@ %.1f fps)',
                frames, fps, fps_avg)
            temp_elo = elo.getRatingList()
            log.info(temp_elo)

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
        temp_elo = elo.getRatingList()
        log.info(temp_elo)

from Algo.arguments import parser
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
flags = parser.parse_args()
eval(flags)