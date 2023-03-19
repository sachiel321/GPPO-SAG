import os.path as osp

import torch
import torch.nn.functional as F
from distar.ctools.utils import deep_merge_dicts, read_config
from .rl_utils import td_lambda_loss,entropy_loss,pg_loss,upgo_loss,kl_loss, ppo_loss

default_config = read_config(osp.join(osp.dirname(__file__), "default_reinforcement_loss.yaml"))

class ReinforcementLoss:
    def __init__(self,) -> None:
        # Here learner_cfg is self._whole_cfg.learner
        self.cfg = default_config.learner

        # policy parameters
        self.gammas = self.cfg.gammas
        ratio = 1.2
        self.ratio_clip = {
            'action_type':ratio, 
            'target_card':ratio,
            'target_entity':ratio, 
            'target_position':ratio, 
        }
        self.ratio_rollback = 0.3

        # loss weight
        self.loss_weights = self.cfg.loss_weights

        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False

    def compute_loss(self, inputs, head_type, factor, factor_teacher):
        # learner compute action_value result
        target_policy_logits_dict = inputs['target_logit']  # shape (T,B)
        value = inputs['value']  # shape (T+1,B)
        dones = inputs['done']  # shape (T,B)
        behaviour_action_log_probs_dict = inputs['action_log_prob']  # shape (T,B)
        teacher_policy_logprobs_dict = inputs['teacher_logprob']  # shape (T,B)

        mask_head_dict = inputs['mask_head']
        masks_dict = inputs['mask']  # shape (T,B)
        actions_dict = inputs['action']  # shape (T,B)
        reward = inputs['reward']  # shape (T,B)

        

        value *= ~dones 

        baseline_value = torch.zeros((value.shape[0]+1,value.shape[1])).to(value.device)
        baseline_value[:-1] = value

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

        actions = actions_dict[head_type].squeeze()
        # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_policy_probs = pi_target.probs
        target_policy_log_probs = pi_target.logits
        target_action_log_probs = pi_target.log_prob(actions)
        
        pi_behavior = torch.distributions.Categorical(logits=behaviour_action_log_probs_dict[head_type])
        behaviour_action_log_probs = pi_behavior.log_prob(actions)

        # behaviour_action_log_probs = behaviour_action_log_probs_dict[head_type]

        target_policy_probs_dict[head_type] = target_policy_probs
        target_policy_log_probs_dict[head_type] = target_policy_log_probs
        target_action_log_probs_dict[head_type] = target_action_log_probs

        teacher_policy_log_probs_dict = {}
        if True:
            pi_teacher = torch.distributions.Categorical(logits=teacher_policy_logprobs_dict[head_type])
            teacher_policy_log_probs = pi_target.logits
            teacher_action_log_probs = pi_teacher.log_prob(actions)
            teacher_policy_log_probs_dict[head_type] = teacher_policy_log_probs

            log_rhos = target_action_log_probs - behaviour_action_log_probs
            log_rhos_teacher_behavior = teacher_action_log_probs - behaviour_action_log_probs
            with torch.no_grad():
                clipped_rhos_vtrace = torch.exp(log_rhos).clamp_(max=1)
            rhos = torch.exp(log_rhos) * factor
            rhos*=~dones
            rhos_teacher_behavior = torch.exp(log_rhos_teacher_behavior) * factor_teacher
            rhos_teacher_behavior*=~dones
            clipped_rhos = rhos.clip(min=rhos_teacher_behavior-self.ratio_clip[head_type],max=rhos_teacher_behavior+self.ratio_clip[head_type])
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

        policy_gradient_loss = \
            ppo_loss(baseline_value, reward, clipped_rhos_dict, clipped_rhos_vtrace, rhos_dict,
                                    masks_dict, head_type=head_type, dones=dones, gamma=1.0)
        
        # policy_gradient_loss = \
        #     pg_loss(baseline_value, reward, dones, target_action_log_probs_dict, clipped_rhos_vtrace,
        #                             masks_dict, head_type=head_type, gamma=1.0)

        total_policy_gradient_loss += self.loss_weights.pg * policy_gradient_loss

        policy_gradient_loss_dict['ppo/'+head_type] = policy_gradient_loss.item()

        loss_info_dict.update(policy_gradient_loss_dict)
        
        # =======
        # kl loss
        # =======
        total_kl_loss = 0

        kl, kl_loss_dict = \
            kl_loss(target_policy_log_probs_dict, teacher_policy_log_probs_dict, masks_dict, head_type=head_type, dones=dones)
        total_kl_loss += kl * self.loss_weights.kl


        loss_info_dict.update(kl_loss_dict)
        
        
        # ===========
        # upgo loss
        # ===========
        total_upgo_loss, upgo_loss_dict = upgo_loss(
            baseline_value, reward, target_action_log_probs_dict, clipped_rhos_dict,
            masks_dict, head_type, dones)

        total_upgo_loss *= self.loss_weights.upgo
        loss_info_dict.update(upgo_loss_dict)

        # ===========
        # critic loss
        # ===========
        total_critic_loss = 0

        #TODO:
        # Notice: in general, we need to include done when we consider discount factor, but in our implementation
        # of alphastar, traj_data(with size equal to unroll-len) sent from actor comes from the same episode.
        # If the game is draw, we don't consider it is actually done
        critic_loss = td_lambda_loss(baseline_value, reward, dones, masks_dict, gamma=self.gammas)

        total_critic_loss += self.loss_weights.baseline * critic_loss
        loss_info_dict['/td'] = critic_loss.item()
        loss_info_dict['/reward'] = reward.float().mean().item()
        loss_info_dict['/value'] = baseline_value.mean().item()
        # ============
        # entropy loss
        # ============
        total_entropy_loss, entropy_dict = \
            entropy_loss(target_policy_probs_dict, target_policy_log_probs_dict, masks_dict,
                         head_type, dones)

        total_entropy_loss *= self.loss_weights.entropy
        loss_info_dict.update(entropy_dict)

        if self.only_update_value:
            total_loss = total_critic_loss/3
        else:
            total_loss = total_policy_gradient_loss + \
                         total_critic_loss/3 + \
                         total_upgo_loss + \
                         total_entropy_loss # + \
                         # total_kl_loss
            
        loss_info_dict['total_loss'] = total_loss
        return loss_info_dict

    def clip_factor(self, inputs, head_type, factor, factor_teacher):
        target_policy_logits_dict = inputs['target_logit']  # shape (T,B)

        behaviour_action_log_probs_dict = inputs['action_log_prob']  # shape (T,B)
        teacher_policy_logprobs_dict = inputs['teacher_logprob']  # shape (T,B)

        masks_dict = inputs['mask']  # shape (T,B)
        actions_dict = inputs['action']  # shape (T,B)
        dones = inputs['done']

        # get distribution info for behaviour policy and target policy
        # take info from correspondent input dict
        target_policy_logits = target_policy_logits_dict[head_type]

        actions = actions_dict[head_type].squeeze()
        # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        # target_policy_probs = pi_target.probs
        # target_policy_log_probs = pi_target.logits
        target_action_log_probs = pi_target.log_prob(actions)

        pi_behavior = torch.distributions.Categorical(logits=behaviour_action_log_probs_dict[head_type])
        behaviour_action_log_probs = pi_behavior.log_prob(actions)
 
        pi_teacher = torch.distributions.Categorical(logits=teacher_policy_logprobs_dict[head_type])
        teacher_action_log_probs = pi_teacher.log_prob(actions)

        log_rhos = target_action_log_probs - behaviour_action_log_probs
        log_rhos_teacher_behavior = teacher_action_log_probs - behaviour_action_log_probs
        
        rhos = torch.exp(log_rhos)
        rhos*=~dones
        rhos_teacher_behavior = torch.exp(log_rhos_teacher_behavior)
        rhos_teacher_behavior*=~dones
        # clipped_rhos = rhos.clip(min=rhos_teacher_behavior-self.ratio_clip[head_type],max=rhos_teacher_behavior+self.ratio_clip[head_type])

        factor_teacher *= rhos_teacher_behavior
        factor *= rhos
        clipped_factor = factor.clip(min=factor_teacher-self.ratio_clip[head_type]/2,max=factor_teacher+self.ratio_clip[head_type]/2)
        # clipped_factor = factor.clip(min=1-self.ratio_clip[head_type]/2,max=1+self.ratio_clip[head_type]/2)
        

        return clipped_factor.detach(), factor.detach(), factor_teacher.detach()



    def reset(self, learner_cfg):
        self.cfg = deep_merge_dicts(self.cfg, learner_cfg)
        # policy parameters
        self.gammas = self.cfg.gammas

        # loss weight
        self.loss_weights = self.cfg.loss_weights
        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False
