import os
import time
import traceback
import uuid
from collections import defaultdict
import torch
import torch.multiprocessing as mp
import random
import json
import platform

from distar.agent.import_helper import import_module
from distar.ctools.utils import read_config, deep_merge_dicts
from distar.ctools.utils.log_helper import TextLogger, VariableRecord
from distar.ctools.worker.actor.actor_comm import ActorComm
from distar.ctools.utils.dist_helper import dist_init
from distar.envs.env import SC2Env
from distar.ctools.worker.league.player import FRAC_ID

default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))


class Actor(object):
    def __init__(self, cfg):
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = cfg.actor.job_type
        self._league_job_type = cfg.actor.get('league_job_type','train')
        self._actor_uid = str(uuid.uuid1())
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)
        print(f'gpu_batch_inference:{self._gpu_batch_inference}')
        self._logger = TextLogger(
            path=os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name, 'actor_log'),
            name=self._actor_uid)
        if self._job_type == 'train':
            self._comm = ActorComm(self._whole_cfg, self._actor_uid, self._logger)
            self.max_job_duration = self._whole_cfg.communication.actor_ask_for_job_interval * random.uniform(0.7 , 1.3)
        self._setup_agents()

    def _setup_agents(self):
        self.agents = []
        if self._job_type == 'train':
            # comm setup agents
            self._comm.ask_for_job(self)
        else:
            self.models = {}
            map_names = []
            for idx, player_id in enumerate(self._cfg.player_ids):
                if 'bot' in player_id:
                    continue
                Agent = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = Agent(self._whole_cfg)
                agent.player_id = player_id
                agent.side_id = idx
                self.agents.append(agent)
                agent.side_id = idx
                if agent.HAS_MODEL:
                    if player_id not in self.models.keys():
                        if self._cfg.use_cuda:
                            assert 'test' in self._job_type, 'only test mode support gpu'
                            agent.model = agent.model.cuda()
                        else:
                            agent.model = agent.model.eval().share_memory()
                        if not self._cfg.fake_model:
                            state_dict = torch.load(self._cfg.model_paths[player_id], map_location='cpu')
                            if 'map_name' in state_dict:
                                map_names.append(state_dict['map_name'])
                                agent._fake_reward_prob = state_dict['fake_reward_prob']
                                agent._z_path = state_dict['z_path']
                                agent.z_idx = state_dict['z_idx']
                            model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                'value_networks' not in k}
                            agent.model.load_state_dict(model_state_dict, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]
            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            if len(map_names) == 2:
                if not(map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'
            if self._job_type == 'train_test':
                teacher_models = {}
                for idx, teacher_player_id in enumerate(self._cfg.teacher_player_ids):
                    if 'bot' in self._cfg.player_ids[idx]:
                        continue
                    agent = self.agents[idx]
                    agent.teacher_player_id = teacher_player_id
                    if agent.HAS_TEACHER_MODEL:
                        if teacher_player_id not in teacher_models.keys():
                            if self._cfg.use_cuda:
                                agent.teacher_model = agent.teacher_model.cuda()
                            else:
                                agent.teacher_model = agent.teacher_model.eval()
                            if not self._cfg.fake_model:
                                state_dict = torch.load(self._cfg.teacher_model_paths[teacher_player_id],
                                                        map_location='cpu')
                                model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                    'value_networks' not in k}
                                agent.teacher_model.load_state_dict(model_state_dict)
                            teacher_models[teacher_player_id] = agent.teacher_model
                        else:
                            agent.teacher_model = teacher_models[teacher_player_id]

    def _inference_loop(self, env_id=0, job={}, result_queue=None, pipe_c=None):
        torch.set_num_threads(1)
        frac_ids = job.get('frac_ids',[])
        env_info = job.get('env_info', {})
        races = []
        for frac_id in frac_ids:
            races.append(random.choice(FRAC_ID[frac_id]))
        if len(races) >0:
            env_info['races']=races
        mergerd_whole_cfg = deep_merge_dicts(self._whole_cfg, {'env': env_info})
        self._env = SC2Env(mergerd_whole_cfg)

        iter_count = 0
        if env_id == 0:
            variable_record = VariableRecord(self._cfg.print_freq)
            variable_record.register_var('agent_time')
            variable_record.register_var('agent_time_per_agent')
            variable_record.register_var('env_time')
            if 'train' in self._job_type:
                variable_record.register_var('post_process_time')
                variable_record.register_var('post_process_per_agent')
                variable_record.register_var('send_data_time')
                variable_record.register_var('send_data_per_agent')
                variable_record.register_var('send_data_per_agent')
                variable_record.register_var('update_model_time')
        with torch.no_grad():
            episode_count = 0
            while episode_count < self._cfg.episode_num:
                try:
                    game_start = time.time()
                    game_iters = 0
                    observations, game_info, map_name = self._env.reset()
                    for idx in observations.keys():
                        self.agents[idx].env_id = env_id
                        race = self._whole_cfg.env.races[idx]
                        self.agents[idx].reset(map_name, race, game_info[idx], observations[idx])

                    while True:  # one episode loop
                        if pipe_c is not None and pipe_c.poll():
                            cmd = pipe_c.recv()
                            if cmd == 'reset':
                                break
                            elif cmd == 'close':
                                self._env.close()
                                return
                        # agent step
                        agent_start_time = time.time()
                        agent_count = 0
                        actions = {}

                        players_obs = observations
                        for player_index, obs in players_obs.items():
                            player_id = self.agents[player_index].player_id
                            if self._job_type == 'train':
                                self.agents[player_index]._model_last_iter = self._comm.model_last_iter_dict[player_id].item()
                            actions[player_index] = self.agents[player_index].step(obs)
                            agent_count += 1
                        agent_time = time.time() - agent_start_time

                        # env step
                        env_start_time = time.time()
                        next_observations, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start_time
                        next_players_obs = next_observations
                        # collect data
                        if 'train' in self._job_type:
                            post_process_time = 0
                            post_process_count = 0
                            send_data_time = 0
                            send_data_count = 0
                            for player_index, obs in next_players_obs.items():
                                if self._job_type == 'train_test' or self.agents[player_index].player_id in self._comm.job[
                                    'send_data_players']:
                                    post_process_start_time = time.time()
                                    traj_data = self.agents[player_index].collect_data(next_players_obs[player_index],
                                                                                    reward[player_index], done, player_index)
                                    post_process_time += time.time() - post_process_start_time
                                    post_process_count += 1
                                    if traj_data is not None and self._job_type == 'train':
                                        send_data_start_time = time.time()
                                        self._comm.send_data(traj_data, self.agents[player_index].player_id)
                                        send_data_time += time.time() - send_data_start_time
                                        send_data_count += 1
                                else:
                                    self.agents[player_index].update_fake_reward(next_players_obs[player_index])

                        # update log
                        iter_count += 1
                        game_iters += 1
                        if env_id == 0:
                            if 'train' in self._job_type:
                                variable_record.update_var(
                                    {'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time,
                                    })
                                if post_process_count > 0:
                                    variable_record.update_var(
                                        {'post_process_time': post_process_time,
                                        'post_process_per_agent': post_process_time / post_process_count,
                                        })
                                if send_data_count > 0:
                                    variable_record.update_var({
                                        'send_data_time': send_data_time,
                                        'send_data_per_agent': send_data_time / send_data_count,
                                    })

                            else:
                                variable_record.update_var({'agent_time': agent_time,
                                                            'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                                            'env_time': env_time, })
                            self.iter_after_hook(iter_count, variable_record)

                        if not done:
                            observations = next_observations
                        else:
                            players_obs = observations
                            if 'test' in self._whole_cfg and self._whole_cfg.test.get('tb_stat', False):
                                if not os.path.exists(self._env._result_dir):
                                    os.makedirs(self._env._result_dir)
                                data = self.agents[0].get_stat_data()
                                path = os.path.join(self._env._result_dir, '{}_{}_{}_.json'.format(env_id, episode_count, player_index))
                                with open(path, 'w') as f:
                                    json.dump(data, f)

                            if self._job_type == 'train':
                                player_idx = random.sample(players_obs.keys(), 1)[0]
                                game_steps = players_obs[player_idx]['raw_obs'].observation.game_loop
                                result_info = defaultdict(dict)

                                for player_index in range(len(self.agents)):
                                    player_id = self.agents[player_index].player_id
                                    side_id = self.agents[player_index].side_id
                                    race = self.agents[player_index].race
                                    agent_iters = self.agents[player_index].iter_count
                                    result_info[side_id]['race'] = race
                                    result_info[side_id]['player_id'] = player_id
                                    result_info[side_id]['opponent_id'] = self.agents[player_index].opponent_id
                                    result_info[side_id]['winloss'] = reward[player_index]
                                    result_info[side_id]['agent_iters'] = agent_iters
                                    result_info[side_id].update(self.agents[player_index].get_unit_num_info())
                                    result_info[side_id].update(self.agents[player_index].get_stat_data())
                                game_duration = time.time() - game_start
                                result_info['game_steps'] = game_steps
                                result_info['game_iters'] = game_iters
                                result_info['game_duration'] = game_duration
                                self._comm.send_result(result_info)
                            break

                    episode_count += 1
                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    episode_count += 1
                    self._env.close()
            self._env.close()
            if result_queue is not None:
                print(os.getpid(), 'done')
                result_queue.put('done')
                time.sleep(1000000)
            else:
                return

    def _gpu_inference_loop(self):
        # _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)
        for agent in self.agents:
            agent.model = agent.model.cuda()
            if 'train' in self._job_type:
                agent.teacher_model = agent.teacher_model.cuda()
        start_time = time.time()
        done_count = 0
        with torch.no_grad():
            while True:
                if self._job_type == 'train':
                    self._comm.async_update_model(self)
                    if time.time() - start_time > self.max_job_duration:
                        self.close()
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self.close()
                            break
                elif self._job_type == 'eval':
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self._close_processes()
                            break
                for agent in self.agents:
                    agent.gpu_batch_inference()
                    if 'train' in self._job_type:
                        agent.gpu_batch_inference(teacher=True)

    def _start_multi_inference_loop(self):
        self._close_processes()
        self._processes = []
        if hasattr(self, '_comm'):
            job = self._comm.job
        else:
            job = {}
        self.pipes = []
        processes = []
        context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        mp_context = mp.get_context(context_str)
        self._result_queue = mp_context.Queue()
        for env_id in range(self._cfg.env_num):
            pipe_p, pipe_c = mp_context.Pipe()
            p = mp_context.Process(target=self._inference_loop, args=(env_id, job, self._result_queue, pipe_c), daemon=True)
            self.pipes.append(pipe_p)
            processes.append(p)
            p.start()
        self.processes = processes

    def reset_env(self):
        for p in self.pipes:
            p.send('reset')

    def run(self):
        try:
            if 'test' in self._job_type:
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        start_time = time.time()
                        while True:
                            if time.time() - start_time > self.max_job_duration:
                                self.reset()
                            self._comm.update_model(self)
                            time.sleep(1)
                if self._job_type == 'eval':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        for _ in range(len(self._processes)):
                            self._result_queue.get()
                        self._close_processes()
        except Exception as e:
            print('[MAIN LOOP ERROR]', e, flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    def reset(self):
        self._logger.info('actor reset multi-process')
        self._close_processes()
        self._comm.ask_for_job(self)
        self._start_multi_inference_loop()

    def close(self):
        self._logger.info('actor close')
        time.sleep(2)
        self._comm.close()
        self._close_processes()
        time.sleep(1)
        os._exit(0)

    def _close_processes(self):
        if hasattr(self, '_processes'):
            print(f'Debug info:len(pipes){len(self.pipes)} | len(_processes){len(self._processes)}')
            for p in self.pipes:
                p.send('close')
            self.pipes = []
            for p in self._processes:
                p.join()

    def iter_after_hook(self, iter_count, variable_record):
        if iter_count % self._cfg.print_freq == 0:
            if hasattr(self,'_comm'):
                variable_record.update_var({'update_model_time':self._comm._avg_update_model_time.item() })
            self._logger.info(
                'ACTOR({}):\n{}TimeStep{}{} {}'.format(
                    self._actor_uid, '=' * 35, iter_count, '=' * 35,
                    variable_record.get_vars_text()
                )
            )


if __name__ == '__main__':
    actor = Actor(cfg={})
    actor.run()
