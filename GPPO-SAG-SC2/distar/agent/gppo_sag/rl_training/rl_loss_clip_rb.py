import os.path as osp

import torch
import torch.nn.functional as F
from distar.ctools.utils import deep_merge_dicts, read_config
from .as_rl_utils import td_lambda_loss,entropy_loss,policy_gradient_loss,upgo_loss,kl_loss,nd_loss, ppo_loss

default_config = read_config(osp.join(osp.dirname(__file__), "default_reinforcement_loss.yaml"))

class ReinforcementLoss:
    def __init__(self, learner_cfg: dict, player_id) -> None:
        # Here learner_cfg is self._whole_cfg.learner
        self.cfg = deep_merge_dicts(default_config.learner,learner_cfg)

        # policy parameters
        self.gammas = self.cfg.gammas
        scale = 0.4
        self.ratio_clip = {
            'action_type':0.1 * scale, 
            'delay':0.15 * scale,
            'queued':0.15 * scale, 
            'target_unit':0.1 * scale, 
            'selected_units':0.3 * scale,
            'target_location':0.2 * scale
        }

        # loss weight
        self.loss_weights = self.cfg.loss_weights
        self.action_type_kl_steps = self.cfg.kl.action_type_kl_steps
        self.dapo_steps = self.cfg.dapo.dapo_steps
        self.use_dapo = self.cfg.use_dapo
        if 'MP' not in player_id:
            self.use_dapo = False
            self.loss_weights.dapo = 0.0
        self.dapo_head_weights = self.cfg.dapo_head_weights
        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False
        self.use_total_rhos = self.cfg.get('use_total_rhos',False)

    def compute_loss(self, inputs, head_type, factor, factor_teacher):
        # learner compute action_value result
        target_policy_logits_dict = inputs['target_logit']  # shape (T,B)
        baseline_values_dict = inputs['value']  # shape (T+1,B)

        behaviour_action_log_probs_dict = inputs['action_log_prob']  # shape (T,B)
        teacher_policy_logits_dict = inputs['teacher_logit']  # shape (T,B)
        if self.use_dapo:
            successive_policy_logits_dict = inputs['successive_logit']
        masks_dict = inputs['mask']  # shape (T,B)
        actions_dict = inputs['action']  # shape (T,B)
        rewards_dict = inputs['reward']  # shape (T,B)
        # dones = inputs['done']  # shape (T,B)
        game_steps = inputs['step']  # shape (T,B) target_action_log_prob
        flag = rewards_dict['winloss'][-1] == 0
        for filed in baseline_values_dict.keys():
            baseline_values_dict[filed][-1] *= flag
        # ===========
        # preparation
        # ===========
        # create loss show dict
        loss_info_dict = {}

        # create preparation info dict
        target_policy_probs_dict = {}
        target_policy_log_probs_dict = {}
        target_action_log_probs_dict = {}
        # log_rhos_dict = {}
        rhos_dict = {}
        clipped_rhos_dict = {}
        # get distribution info for behaviour policy and target policy

        # take info from correspondent input dict
        target_policy_logits = target_policy_logits_dict[head_type]

        actions = actions_dict[head_type]
        # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_policy_probs = pi_target.probs
        target_policy_log_probs = pi_target.logits
        target_action_log_probs = pi_target.log_prob(actions)

        behaviour_action_log_probs = behaviour_action_log_probs_dict[head_type]

        target_policy_probs_dict[head_type] = target_policy_probs
        target_policy_log_probs_dict[head_type] = target_policy_log_probs
        if head_type == 'selected_units':
            target_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
            target_action_log_probs = target_action_log_probs.sum(-1)
            behaviour_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
            behaviour_action_log_probs = behaviour_action_log_probs.sum(-1)
        target_action_log_probs_dict[head_type] = target_action_log_probs

        # =======
        # kl loss
        # =======
        with torch.no_grad():
            kl, action_type_kl_loss, kl_loss_dict = \
                kl_loss(target_policy_log_probs_dict, teacher_policy_logits_dict, masks_dict, game_steps,
                        action_type_kl_steps=self.action_type_kl_steps, head_type=head_type)
            # total_kl_loss *= self.loss_weights.kl
            action_type_kl_loss *= self.loss_weights.action_type_kl


        loss_info_dict.update(kl_loss_dict)

        if True:
            teacher_policy_logits = teacher_policy_logits_dict[head_type]
            pi_teacher = torch.distributions.Categorical(logits=teacher_policy_logits)
            teacher_action_log_probs = pi_teacher.log_prob(actions).detach()

            if head_type == 'selected_units':
                teacher_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
                teacher_action_log_probs = teacher_action_log_probs.sum(-1)

            log_rhos = target_action_log_probs - behaviour_action_log_probs
            log_rhos_teacher_behavior = teacher_action_log_probs - behaviour_action_log_probs
            with torch.no_grad():
                clipped_rhos_vtrace = torch.exp(log_rhos).clamp_(max=1)
            rhos = torch.exp(log_rhos) * factor
            rhos_teacher_behavior = torch.exp(log_rhos_teacher_behavior) * factor_teacher
            # GPPO-SAG
            clipped_rhos = rhos.clip(min=rhos_teacher_behavior-self.ratio_clip[head_type],max=rhos_teacher_behavior+self.ratio_clip[head_type])
            # PPO
            # clipped_rhos = rhos.clip(min=1-self.ratio_clip[head_type],max=1+self.ratio_clip[head_type])

        # save preparation results to correspondent dict
        
        # log_rhos_dict[head_type] = log_rhos
        clipped_rhos_dict[head_type] = clipped_rhos
        rhos_dict[head_type] = rhos

        loss_info_dict['rho/' + head_type] = abs(rhos).max().item()

        # ====================
        # policy gradient loss
        # ====================
        total_policy_gradient_loss = 0
        policy_gradient_loss_dict = {}

        for field, baseline in baseline_values_dict.items():
            baseline_value = baseline_values_dict[field]
            reward = rewards_dict[field]
            # PG loss
            # field_policy_gradient_loss = \
            #     policy_gradient_loss(baseline_value, reward, target_action_log_probs_dict, clipped_rhos_dict,
            #                          masks_dict, head_type=head_type, factor=factor, gamma=1.0, field=field)
            
            # PPO loss
            field_policy_gradient_loss = \
                ppo_loss(baseline_value, reward, clipped_rhos_dict, clipped_rhos_vtrace, rhos_dict,
                                     masks_dict, head_type=head_type, gamma=1.0, field=field)

            total_policy_gradient_loss += self.loss_weights.pg[field] * field_policy_gradient_loss

            policy_gradient_loss_dict[field + '/' + head_type] = field_policy_gradient_loss.item()

        loss_info_dict.update(policy_gradient_loss_dict)
        # ===========
        # upgo loss
        # ===========
        total_upgo_loss, upgo_loss_dict = upgo_loss(
            baseline_values_dict['winloss'], rewards_dict['winloss'], target_action_log_probs_dict, clipped_rhos_dict,
            masks_dict['actions_mask'], head_type)

        total_upgo_loss *= self.loss_weights.upgo.winloss
        loss_info_dict.update(upgo_loss_dict)

        # ===========
        # critic loss
        # ===========
        total_critic_loss = 0

        # field is from ['winloss', 'build_order','built_unit','effect','upgrade','battle']
        for field, baseline in baseline_values_dict.items():
            reward = rewards_dict[field]
            # td_lambda_loss = self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]

            # Notice: in general, we need to include done when we consider discount factor, but in our implementation
            # of alphastar, traj_data(with size equal to unroll-len) sent from actor comes from the same episode.
            # If the game is draw, we don't consider it is actually done
            critic_loss = td_lambda_loss(baseline, reward, masks_dict, gamma=self.gammas.baseline[field], field=field)

            total_critic_loss += self.loss_weights.baseline[field] * critic_loss
            loss_info_dict[field + '/td'] = critic_loss.item()
            loss_info_dict[field + '/reward'] = reward.float().mean().item()
            loss_info_dict[field + '/value'] = baseline.mean().item()
        loss_info_dict['battle' + '/reward'] = rewards_dict['battle'].float().mean().item()
        # ============
        # entropy loss
        # ============
        total_entropy_loss, entropy_dict = \
            entropy_loss(target_policy_probs_dict, target_policy_log_probs_dict, masks_dict,
                         head_type)

        total_entropy_loss *= self.loss_weights.entropy
        loss_info_dict.update(entropy_dict)

        

        # # =========
        # # DAPO loss
        # # =========
        # if self.use_dapo:
        #     total_p3d_loss, p3d_loss_dict = \
        #         penalized_point_probability_distance_loss(target_policy_log_probs_dict, successive_policy_logits_dict, masks_dict, game_steps,
        #                 dapo_steps=self.dapo_steps, head_type=head_type) 
        #     total_p3d_loss *= self.loss_weights.dapo
        #     loss_info_dict.update(p3d_loss_dict)
        # else:
        #     total_p3d_loss = 0.0

        if self.only_update_value:
            total_loss = total_critic_loss/6
        else:
            total_loss = total_policy_gradient_loss + \
                         total_critic_loss/6 + \
                         total_upgo_loss + \
                         total_entropy_loss
            
        loss_info_dict['total_loss'] = total_loss
        return loss_info_dict

    def clip_factor(self, inputs, head_type, factor, factor_teacher):
        target_policy_logits_dict = inputs['target_logit']  # shape (T,B)

        behaviour_action_log_probs_dict = inputs['action_log_prob']  # shape (T,B)
        teacher_policy_logits_dict = inputs['teacher_logit']  # shape (T,B)

        masks_dict = inputs['mask']  # shape (T,B)
        actions_dict = inputs['action']  # shape (T,B)

        # get distribution info for behaviour policy and target policy
        # take info from correspondent input dict
        target_policy_logits = target_policy_logits_dict[head_type]

        actions = actions_dict[head_type]
        # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        # target_policy_probs = pi_target.probs
        # target_policy_log_probs = pi_target.logits
        target_action_log_probs = pi_target.log_prob(actions)

        behaviour_action_log_probs = behaviour_action_log_probs_dict[head_type]

        if head_type == 'selected_units':
            target_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
            target_action_log_probs = target_action_log_probs.sum(-1)
            behaviour_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
            behaviour_action_log_probs = behaviour_action_log_probs.sum(-1)
 
        teacher_policy_logits = teacher_policy_logits_dict[head_type]
        pi_teacher = torch.distributions.Categorical(logits=teacher_policy_logits)
        teacher_action_log_probs = pi_teacher.log_prob(actions).detach()

        if head_type == 'selected_units':
            teacher_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
            teacher_action_log_probs = teacher_action_log_probs.sum(-1)

        log_rhos = target_action_log_probs - behaviour_action_log_probs
        log_rhos_teacher_behavior = teacher_action_log_probs - behaviour_action_log_probs
        
        rhos = torch.exp(log_rhos)
        rhos_teacher_behavior = torch.exp(log_rhos_teacher_behavior)

        factor_teacher *= rhos_teacher_behavior
        factor *= rhos
        # GPPO-SAG's double clip
        clipped_factor = factor.clip(min=factor_teacher-self.ratio_clip[head_type]*1.3333,max=factor_teacher+self.ratio_clip[head_type]*1.3333)
        

        return clipped_factor.detach(), factor.detach(), factor_teacher.detach()



    def reset(self, learner_cfg):
        self.cfg = deep_merge_dicts(self.cfg, learner_cfg)
        # policy parameters
        self.gammas = self.cfg.gammas

        # loss weight
        self.loss_weights = self.cfg.loss_weights
        self.action_type_kl_steps = self.cfg.kl.action_type_kl_steps
        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False
