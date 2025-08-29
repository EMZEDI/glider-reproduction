import deepspeed
import pandas as pd
import random
import torch
from util.model import Policy
from alg.bc import Agent
from scienceworld import ScienceWorldEnv
import copy
from prompt.inst import high_prompt, low_prompt,subtask_complete_prompt
from util.extract import extract_action_done
from util.replay_buffer import batch_traj_process
from torch.nn.utils.rnn import pad_sequence

class EvalAgent:
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=hierarcy_policy.parameters(),
                                                     config=args["ds_config"])
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

    def load_policy(self, path):
        Agent.load_policy(self, path)

    def load_critic(self):
        """Load the critic weights for Q-value analysis"""
        import os
        critic_path = os.path.join(self.checkpoint_dir, "critic.pth")
        if os.path.exists(critic_path):
            # Add critic to the policy if it doesn't exist
            if not hasattr(self.engine.module, 'critic'):
                hidden_dim = self.engine.module.base.config.hidden_size
                import torch.nn as nn
                self.engine.module.critic = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            
            critic_state_dict = torch.load(critic_path, map_location=self.engine.device)
            self.engine.module.critic.load_state_dict(critic_state_dict)
            print(f"Loaded critic from {critic_path}")
        else:
            print(f"Warning: Critic not found at {critic_path}")

    @staticmethod
    def _discounted_returns_list(rewards_list, dones_list, gamma: float):
        """Compute Monte Carlo returns for each episode"""
        out = []
        for r_seq, d_seq in zip(rewards_list, dones_list):
            T = len(r_seq)
            G = torch.zeros(T, dtype=torch.float32)
            running = 0.0
            for t in range(T - 1, -1, -1):
                running = float(r_seq[t]) + gamma * running * (1.0 - float(d_seq[t]))
                G[t] = running
            out.append(G)
        return out

    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2:
            return 0.0
        x = x.float() - x.float().mean()
        y = y.float() - y.float().mean()
        denom = (x.norm() * y.norm()).clamp(min=1e-8)
        return float((x * y).sum() / denom)

    @staticmethod
    def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2:
            return 0.0
        def rankify(v: torch.Tensor) -> torch.Tensor:
            order = torch.argsort(v)
            ranks = torch.zeros_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(1, v.numel() + 1, device=v.device, dtype=torch.float32)
            return ranks
        rx = rankify(x) - rankify(x).mean()
        ry = rankify(y) - rankify(y).mean()
        denom = (rx.norm() * ry.norm()).clamp(min=1e-8)
        return float((rx * ry).sum() / denom)

    def extract_valid(self, value, valid_mark):
        """Extract valid values using the same logic as training code"""
        batch_size = value.size(0)
        max_valid_len = valid_mark.sum(dim=1).max().item()

        valid_value = torch.zeros(batch_size, max_valid_len, device=value.device)
        mask = torch.zeros(batch_size, max_valid_len, device=value.device)
        for i in range(batch_size):
            valid_idx = torch.where(valid_mark[i] == 1)[0]
            valid_len = valid_idx.size(0)
            
            valid_value[i, :valid_len] = value[i][valid_idx]
            mask[i, :valid_len] = 1

        return valid_value, mask

    @torch.no_grad()
    def evaluate_q_mc_from_episodes(self, episodes: dict):
        """Compare critic Q(s_t, a_t) against Monte Carlo returns"""
        if not hasattr(self.engine.module, 'critic'):
            print("Warning: No critic found, skipping Q-value analysis")
            return {}

        self.engine.module.eval()

        # Get critic predictions for high-level actions
        batch_tokens = batch_traj_process(
            episodes['task_description'],
            episodes['obs'],
            episodes['subtask'], 
            self.engine.tokenizer
        ).to(self.engine.device)

        hidden_states, _, action_end_mask = self.engine.get_hidden_states(batch_tokens)
        q_seq = self.engine.module.critic(hidden_states).squeeze(-1)  # (B, L)
        q_sa, mask = self.extract_valid(q_seq, action_end_mask)  # (B, T_pad)

        # Compute Monte Carlo returns
        gamma = self.args.get('gama', 0.99)  # Use same gamma as training
        G_list = self._discounted_returns_list(episodes['reward'], episodes['done'], gamma)
        G = pad_sequence([g.to(self.engine.device) for g in G_list], batch_first=True, padding_value=0.0)

        # Compare only at valid positions
        valid = mask.bool()
        qv = q_sa[valid]  # Critic Q estimates
        gv = G[valid]     # True MC returns

        # Compute metrics
        diff = qv - gv
        mse = float(diff.pow(2).mean())
        mae = float(diff.abs().mean()) 
        bias = float(diff.mean())
        pearson = self._pearson_corr(qv, gv)
        spearman = self._spearman_corr(qv, gv)

        metrics = {
            "mse": mse,
            "mae": mae,
            "bias": bias, 
            "pearson_r": pearson,
            "spearman_r": spearman,
            "n_steps": int(valid.sum().item()),
            "q_mean": float(qv.mean()),
            "g_mean": float(gv.mean()),
            "q_std": float(qv.std()) if qv.numel() > 1 else 0.0,
            "g_std": float(gv.std()) if gv.numel() > 1 else 0.0
        }

        return metrics

    def eval(self, dev_or_test):
        self.load_policy(self.checkpoint_dir)
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')[f'{dev_or_test}'].tolist()
        task_score = {}
        for task_id, vari_count in enumerate(vari_nums):
            task_name = self.task_names[task_id]
            self.eval_env.load(task_name)
            task_score[task_name] = []
            if dev_or_test == "test":
                vari_ids = self.eval_env.getVariationsTest()
            elif dev_or_test == "dev":
                vari_ids = self.eval_env.getVariationsDev()
            else:
                vari_ids = list(range(vari_count))
            
            for vari_id in random.sample(vari_ids, vari_count):
                score = self.eval_policy(task_id, vari_id)
                # task_reward[task_name].append(reward)
                task_score[task_name].append(score)
            print("task_score: ", task_score)


        average_score= []
        for _, value in task_score.items():
            if len(value):
                average_score.append(sum(value)/len(value))
        print("result: ", sum(average_score)/len(average_score))

    def eval_with_q_analysis(self, dev_or_test, num_episodes_per_task=5):
        """Enhanced evaluation that collects episodes for Q-value analysis"""
        self.load_policy(self.checkpoint_dir)
        self.load_critic()
        
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')[f'{dev_or_test}'].tolist()
        task_score = {}
        
        # Collect episodes for Q-analysis
        all_episodes = {
            'task_description': [],
            'obs': [],
            'subtask': [], 
            'reward': [],
            'done': []
        }

        for task_id, vari_count in enumerate(vari_nums):
            task_name = self.task_names[task_id]
            self.eval_env.load(task_name)
            task_score[task_name] = []
            
            if dev_or_test == "test":
                vari_ids = self.eval_env.getVariationsTest()
            elif dev_or_test == "dev":
                vari_ids = self.eval_env.getVariationsDev()
            else:
                vari_ids = list(range(vari_count))
            
            # Sample episodes for both scoring and Q-analysis
            selected_varis = random.sample(vari_ids, min(num_episodes_per_task, len(vari_ids)))
            
            for vari_id in selected_varis:
                score, episode_data = self.eval_policy_with_data_collection(task_id, vari_id)
                task_score[task_name].append(score)
                
                # Add to episodes for Q-analysis
                all_episodes['task_description'].append(episode_data['task_description'])
                all_episodes['obs'].append(episode_data['obs'])
                all_episodes['subtask'].append(episode_data['subtask'])
                all_episodes['reward'].append(episode_data['reward'])
                all_episodes['done'].append(episode_data['done'])
            
            print(f"Task {task_name}: {task_score[task_name]}")

        # Compute average scores
        average_score = []
        for _, value in task_score.items():
            if len(value):
                average_score.append(sum(value)/len(value))
        
        final_score = sum(average_score)/len(average_score) if average_score else 0.0
        print(f"Final average score: {final_score}")

        # Analyze Q-values
        print("\n=== Q-Value Analysis ===")
        q_metrics = self.evaluate_q_mc_from_episodes(all_episodes)
        if q_metrics:
            print(f"Q-Value MSE: {q_metrics['mse']:.4f}")
            print(f"Q-Value MAE: {q_metrics['mae']:.4f}")
            print(f"Q-Value Bias: {q_metrics['bias']:.4f}")
            print(f"Pearson correlation: {q_metrics['pearson_r']:.4f}")
            print(f"Spearman correlation: {q_metrics['spearman_r']:.4f}")
            print(f"Steps analyzed: {q_metrics['n_steps']}")
            print(f"Q mean/std: {q_metrics['q_mean']:.3f} ± {q_metrics['q_std']:.3f}")
            print(f"MC mean/std: {q_metrics['g_mean']:.3f} ± {q_metrics['g_std']:.3f}")

        return final_score, q_metrics

    def eval_policy(self, task_id, vari_id):
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        obs, _= self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        print(f"task:{task_name}, vari:{vari_id}, {task_description}")
        # high: (high_prompt, task_description, cache_obs0, subtask_0...cache_obst)->subtask_t
        high_traj_token = self.engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')
        traj_subtask, traj_group_action = [], []
        group_action = []
        done = False
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            print("subtask:", subtask)
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            traj_subtask.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            raw_action_list = []
            group_reward, group_score = 0.0, 0.0
            
            while not subtask_done:
                episode_steps += 1
                # (low_prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
                obs_token = self.engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward
                group_score += info['score']
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            traj_group_action.append(group_action)
            print("group action: ", raw_action_list)

        # print("subtask: ", traj_subtask)
        # print("group action:", traj_group_action)
        score = max(0, info['score'])
        print(f"score: {score}")
        return score

    def eval_policy_with_data_collection(self, task_id, vari_id):
        """Modified eval_policy that also collects episode data for Q-analysis"""
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        obs, _= self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        print(f"task:{task_name}, vari:{vari_id}, {task_description}")
        
        # Episode data collection
        episode_data = {
            'task_description': high_prompt + " " + task_description,
            'obs': [],
            'subtask': [],
            'reward': [],
            'done': []
        }
        
        high_traj_token = self.engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')
        done = False
        group_action = []
        
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            episode_data['obs'].append(state)
            
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            print("subtask:", subtask)
            
            episode_data['subtask'].append(subtask)
            
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            raw_action_list = []
            group_reward = 0.0
            
            while not subtask_done:
                episode_steps += 1
                obs_token = self.engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward / 100.0  # Normalize like in training
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            
            episode_data['reward'].append(group_reward)
            episode_data['done'].append(done)
            print("group action: ", raw_action_list)

        # Add final observation
        final_state = f"Group action: {group_action}. Current observation: {obs}"
        episode_data['obs'].append(final_state)
        
        score = max(0, info['score'])
        print(f"score: {score}")
        return score, episode_data
    
    def data_collect(self, task_id, vari_id, high_data_container, low_data_container):
        """
        Args:
            high_data_container:{
                'task_description': [task_num,],
                'obs': [task_num, groups+1],
                'subtask': [task_num, groups],
                'reward': [task_num, groups],
                'score': [task_num, groups],
                'done': [task_num, groups]
            }
            low_data_container:{
                'subtask':[subtask_nums, steps],
                'obs':[subtask_nums, steps+1],
                'action':[subtask_nums, steps],
                'reward':[subtask_nums, steps],
                'score':[subtask_nums, steps],
                'done':[subtask_nums, steps]
            }
            score_threshold:[min, max]
        """
        high_obs_traj, high_subtask_traj, high_reward_traj, high_score_traj, high_done_traj = [], [], [], [], []
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        task_description = self.eval_env.taskdescription()
        print(task_id, vari_id, task_description)
        obs, _= self.eval_env.reset()
        high_traj_token = self.engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')

        done = False
        group_action = []
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            high_obs_traj.append(state)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            print("subtask:", subtask)
            high_subtask_traj.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            group_reward, group_score = 0.0, 0.0
            raw_action_list = []
            low_obs_traj, low_reward_traj, low_score_traj, low_done_traj = [], [], [], []
            low_init_obs = obs
            while not subtask_done:
                low_obs_traj.append("Obs: "+obs)
                episode_steps += 1
                # (low_prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
                obs_token = self.engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                low_done_traj.append(subtask_done)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward/100
                group_score += info['score']/100
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            print("group action: ", raw_action_list, info['score'])
            # is_subtask_complete_prompt = subtask_complete_prompt.replace("[subtask]",subtask)\
            #                                              .replace("[initial_obs]", low_init_obs)\
            #                                              .replace("[final_obs]", obs)\
            #                                              .replace("[action_sequence]", str(group_action))
            # subtask_complete_token = self.engine.tokenizer(is_subtask_complete_prompt, return_tensors='pt')
            # subtask_complete = self.engine.generate_action(subtask_complete_token)
            # print("subtask_complete:",subtask_complete)
            
            low_obs_traj.append("Obs: "+obs)
            low_data_container['subtask'].append(low_prompt + " Subtask: " + subtask)
            low_data_container['obs'].append(low_obs_traj)
            low_data_container['action'].append(raw_action_list)
            low_data_container['done'].append(low_done_traj)
            
            high_reward_traj.append(group_reward)
            high_score_traj.append(group_score)
            high_done_traj.append(False if episode_steps==self.args['env_step_limit'] else done)
        state = f"Group action: {group_action}. Current observation: {obs}"
        high_obs_traj.append(state)
        high_data_container['task_description'].append(high_prompt + " " + task_description)
        high_data_container['obs'].append(high_obs_traj)
        high_data_container['subtask'].append(high_subtask_traj)
        high_data_container['done'].append(high_done_traj)
        high_data_container['score'].append(high_score_traj)
        high_data_container['reward'].append(high_reward_traj)


    