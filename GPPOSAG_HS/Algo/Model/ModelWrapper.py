from Algo.Model.Cardsformer import Cardsformer, Cardsformer_naive
from StoneZeroModel.PolicyModel import PolicyModel
import torch
import numpy as np
from Env.Hearthstone import log

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512, device=0, vs_stone_zero=True):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['Player1'] = Cardsformer().to(torch.device(device))
        self.models['Player1'].device = torch.device(device)
        self.models['Player2'] = None
        self.models['teacher'] = Cardsformer().to(torch.device(device))
        #TODO: load self.models['teacher'] param
        checkpoint_states = torch.load("StoneZeroPretrainedModels/sl_pretrain/Player1_weights_120000000.0.ckpt", map_location=device)
        model_dict = self.models['Player1'].state_dict()
        param_dict = {k:v for k,v in checkpoint_states.items()}
        model_dict.update(param_dict)
        self.models['Player1'].load_state_dict(model_dict)
        self.models['teacher'].load_state_dict(model_dict)      
        del checkpoint_states, model_dict, param_dict
        
        self.models['teacher'].eval()
        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.models['Player1']])
        log.info('\nNumber of Cardsformer parameters: \t pi: %d'%var_counts)
        if vs_stone_zero:
            checkpoint_states = torch.load('StoneZeroPretrainedModels/Cardsformer_Trained_weights_3000000.ckpt', map_location=device)
            self.stone_zero = PolicyModel(64, 768, 256, 512).to(torch.device(device))
            self.stone_zero.load_state_dict(checkpoint_states)
            self.stone_zero.eval()
        else:
            self.stone_zero = None
        self.lazy_frames = 0
        self.teacher_update_period = 1e6

    def forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, mask, actor):
        model = self.models['Player1']
        return model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar"], obs["minion_scalar"], obs["hero_scalar"], num_options, mask, actor)
    
    def select_forward(self,model_type, hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor):
        if model_type == 'teacher':
            model = self.models['teacher']
        else:
            model = self.models['Player1']
        return model.select_forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar"], obs["minion_scalar"], obs["hero_scalar"], num_options, available_actions, mask, actor)
    
    def kl_estimate(self,model_type, hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor):
        action_info_teacher, logit_teacher, mask_teacher, mask_head_teacher = self.models['teacher'].select_forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar"], obs["minion_scalar"], obs["hero_scalar"], num_options, available_actions, mask, actor)
    
        action_info_target, logit_target, mask_target, mask_head_target = self.models['Player1'].select_forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar"], obs["minion_scalar"], obs["hero_scalar"], num_options, available_actions, mask, actor)

        kl_dict = {}
        for head_type in ['action_type', 'target_entity', 'target_position']:
            actions_target = action_info_target[head_type].squeeze()
            # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
            pi_target = torch.distributions.Categorical(logits=logit_target[head_type])
            target_policy_probs = pi_target.probs
            target_policy_log_probs = pi_target.logits
            
            actions_teacher = action_info_teacher[head_type].squeeze()
            # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
            pi_teacher = torch.distributions.Categorical(logits=logit_teacher[head_type])
            teacher_policy_probs = pi_teacher.probs
            teacher_policy_log_probs = pi_teacher.logits
            
            kl = teacher_policy_probs * (teacher_policy_log_probs - target_policy_log_probs)
            kl = kl.sum(dim=-1).cpu()
            kl *= mask_target[head_type].cpu()
            kl_dict[head_type] = kl
        return action_info_target, kl_dict
        
    
    def stone_zero_select_options(self, card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor):
        return self.stone_zero(card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar_batch"], obs["minion_scalar_batch"], obs["hero_scalar_batch"], obs["next_minion_scalar"], obs["next_hero_scalar"], num_options, actor)
    
    def train_forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand, minions, heros, behaviour_logp, teacher_logit, reward, mask, action_info):
        model = self.models['Player1']
        return model.train_forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, hand, minions, heros, behaviour_logp, teacher_logit, reward, mask, action_info)
    
    def teacher_forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, available_actions, mask, actor):
        model = self.models['teacher']
        return model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar"], obs["minion_scalar"], obs["hero_scalar"], num_options, available_actions, mask, actor)

    def check_teacher_update(self, frames):
        if frames - self.lazy_frames > self.teacher_update_period:
            self.lazy_frames = frames
            self.models['teacher'].load_state_dict(self.models['Player1'].state_dict())
        else:
            pass
    
    def share_memory(self):
        self.models['Player1'].share_memory()
        # self.models['Player2'].share_memory()
        return

    def eval(self):
        self.models['Player1'].eval()
        # self.models['Player2'].eval()

    def parameters(self, position):
        return self.models['Player1'].parameters()

    def get_model(self, position):
        return self.models['Player1']

    def get_models(self):
        return self.models


class Model_naive:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['Player1'] = Cardsformer_naive(card_dim, bert_dim, embed_dim, dim_ff).to(torch.device(device))
        self.models['Player1'].device = torch.device(device)
        self.models['Player2'] = None

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.models['Player1']])
        log.info('\nNumber of Cardsformer parameters: \t pi: %d'%var_counts)

    def forward(self, position, card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor):
        model = self.models['Player1']
        return model.forward(card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_batch"], obs["minions_batch"], obs["heros_batch"], num_options, actor)

    def share_memory(self):
        self.models['Player1'].share_memory()
        # self.models['Player2'].share_memory()
        return

    def eval(self):
        self.models['Player1'].eval()
        # self.models['Player2'].eval()

    def parameters(self, position):
        return self.models['Player1'].parameters()

    def get_model(self, position):
        return self.models['Player1']

    def get_models(self):
        return self.models
