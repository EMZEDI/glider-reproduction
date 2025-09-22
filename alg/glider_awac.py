from util.model import Policy, Critic
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import HierarchyDataset, batch_traj_process
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import deepspeed
import torch
import copy
import os


class ActorCritic(Policy):
    def __init__(self, args):
        super().__init__(args)
        hidden_dim = self.base.config.hidden_size
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.soft_update_target_critic(tau=1.0)

    def soft_update_target_critic(self, tau):
        assert 0.0 <= tau <= 1.0
        for target_param, param in zip(self.target_critic.parameters(), 
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
    
    def critic_forward(self, x):
        return self.critic(x).squeeze(-1)
    
    @torch.no_grad()
    def target_critic_forward(self, x):
        return self.target_critic(x).squeeze(-1)
            
class GLIDER:
    def __init__(self, args):
        self.args = args
        actor_critic = ActorCritic(args)
        self.engine, _ , _, _ = deepspeed.initialize(
                                           model=actor_critic,
                                           model_parameters=[{"params": [p for p in actor_critic.base.parameters() if p.requires_grad], 
                                                              "lr": args["actor_lr"]},
                                                              {"params": [p for p in actor_critic.critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]}],
                                           config=args["ds_config"])
        
        
        path = f"{args['check_path']}/{args['benchmark']}/glider_bc/{args['model_name']}"
        BC_AGENT.load_policy(self, path)
        
        self.loss_fct = torch.nn.MSELoss()
        self.buffer = HierarchyDataset(args)

        if self.engine.global_rank == 0:
            wandb.init(
                project=f"{args['benchmark']}-{args['alg_name']}",
                name=f"{args['model_name']}",
                config=args
            )

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        
        # Add trajectory logging configuration
        self.log_interval = args.get('trajectory_log_interval', 100)  # Log every N steps
        self.max_trajectories_to_log = args.get('max_trajectories_to_log', 3)  # Max trajectories per log
    
    def save_critic(self):
        model_path = os.path.join(self.checkpoint_dir, "critic.pth")
        torch.save(self.engine.module.critic.state_dict(), model_path)
    
    def update_ac(self, batch_data):
        batch_tokens = batch_traj_process(batch_data['task_description'],
                                          batch_data['obs'],
                                          batch_data['subtask'],
                                          self.engine.tokenizer).to(self.engine.device)
        rewards, dones = self.prepare_tensor(batch_data['reward'], batch_data['done'])

        # Critic
        with torch.no_grad():
            hidden_states, _, action_end_mask = self.engine.get_hidden_states(batch_tokens)
            target_qsa = self.engine.target_critic_forward(hidden_states) # (batch, seq_len)
            target_qsa, _ = self.extract_valid(target_qsa, action_end_mask) # (batch, num_action)

        q_sa = self.engine.critic_forward(hidden_states)
        q_sa, _ = self.extract_valid(q_sa, action_end_mask) # (batch, num_action)

        # L_Q
        target = rewards + (1-dones) * F.pad(target_qsa[:, 1:], (0, 1), value=0) * self.args['gama']    # (batch, num_action)
        q_loss = self.loss_fct(q_sa, target)
        self.engine.backward(q_loss)
        self.engine.step()
        
        # Actor
        action_log_probs, action_masks = self.engine.get_log_prob(batch_tokens)
        # valid_action_log_probs,_ = self.extract_valid(action_log_probs, action_end_mask[:, 1:])
        valid_action_log_probs = BC_AGENT.extract_valid_action_probs(self, action_log_probs, action_masks, q_sa.size(1))
       
        actor_loss = -torch.mean(target_qsa.detach()*valid_action_log_probs)
        self.engine.backward(actor_loss)
        self.engine.step()
        return q_loss, actor_loss
            

    def update(self,):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        
        for epoch in range(self.args['epochs']):
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,], or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],       or "obs":[group_nums, steps+1]
                        "subtask":[traj_nums, steps],     or "action":[group_nums, steps]
                        "reward": [traj_nums, steps]      or "reward":[group_nums, steps]
                        "done": [traj_nums, steps]        or "done": [grop_nums, steps]
                    }
                Update Critic:
                    L_Q(\phi) = E_{s,a,r,s'~D}[Q_{\phi}(s,a) - r - \gamma * V_{\bar{\psi}(s')]
                    L_V(\psi) = E_{s~D}[E_{a~\pi_\theta(·|s)}[V_\psi(s)-Q_{\bar{\theta}}(s,a)]]
                Update Actor:
                    L_\pai(\theta) = -E_{s,a~D}[exp(1/lammda * A(s,a)) * log\pai_\theta(a|s)]
                """
                
                # Log trajectories to WandB every N steps
                if (self.engine.local_rank == 0 and 
                    self.global_step.item() % self.log_interval == 0):
                    self.log_trajectories_to_wandb(batch, self.global_step.item())
                
                # high level
                expert_q_loss, expert_actor_loss = self.update_ac(batch['high'])
                medium_q_loss, medium_actor_loss = self.update_ac(batch['medium'])
               
                self.engine.soft_update_target_critic(tau=self.args['tau'])
            
                # low level
                batch_low_tokens = batch_traj_process(batch['low']['subtask'],
                                                      batch['low']['obs'],
                                                      batch['low']['action'],
                                                      self.engine.tokenizer).to(self.engine.device)
                low_log_probs, low_masks = self.engine.get_log_prob(batch_low_tokens)
                low_valid_log_prob = BC_AGENT.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                         max(batch_low_tokens['action_end_mask'].sum(dim=1)))
                low_loss = -low_valid_log_prob.mean()
                self.engine.backward(low_loss)
                self.engine.step()
                
                
                if self.engine.local_rank == 0:  # Only log on the main process
                    wandb.log({
                        'Loss/expert/q_loss': expert_q_loss.item(),
                        'Loss/expert/actor_loss': expert_actor_loss.item(),
                        'Loss/medium/q_loss': medium_q_loss.item(),
                        'Loss/medium/actor_loss': medium_actor_loss.item(),
                        'Loss/low/actor_loss': low_loss.item()
                    }, step=self.global_step.item())
                    print(f"expert; step:{self.global_step.item()}; actor_loss:{expert_actor_loss.item()}; critic_loss:{expert_q_loss.item()}")
                    print(f"medium; step:{self.global_step.item()}; actor_loss:{medium_actor_loss.item()}; critic_loss:{medium_q_loss.item()}")
                    print(f"low; step:{self.global_step.item()}; loss:{low_loss.item()}")

                if self.global_step.item() % self.args['eval_freq'] == 0:
                    BC_AGENT.save_policy(self)
                    self.save_critic()
                self.global_step += 1
        BC_AGENT.save_policy(self)
        self.save_critic()

    def _log_qvalue_analysis(self, expert_data, medium_data, step):
        """Log Q-value vs Monte Carlo return comparison"""
        try:
            for level_name, data in [("expert", expert_data), ("medium", medium_data)]:
                if 'reward' in data and 'obs' in data:
                    # Compute Monte Carlo returns
                    mc_returns = self._compute_monte_carlo_returns(data['reward'])
                    
                    # Get Q-value predictions from critic
                    q_predictions = self._get_q_value_predictions(data)
                    
                    if mc_returns is not None and q_predictions is not None:
                        # Compute metrics
                        td_errors = torch.abs(q_predictions - mc_returns)
                        mse_loss = torch.mean((q_predictions - mc_returns) ** 2)
                        mae_loss = torch.mean(td_errors)
                        
                        wandb.log({
                            f"q_analysis/{level_name}/td_error_mean": td_errors.mean().item(),
                            f"q_analysis/{level_name}/td_error_std": td_errors.std().item(),
                            f"q_analysis/{level_name}/mse_loss": mse_loss.item(),
                            f"q_analysis/{level_name}/mae_loss": mae_loss.item(),
                            f"q_analysis/{level_name}/q_mean": q_predictions.mean().item(),
                            f"q_analysis/{level_name}/mc_mean": mc_returns.mean().item(),
                        }, step=step)
                        
        except Exception as e:
            print(f"Warning: Failed to compute Q-value analysis: {e}")

    def _compute_monte_carlo_returns(self, rewards):
        """Compute Monte Carlo returns (discounted cumulative rewards)"""
        try:
            gamma = self.args.get('gama', 0.99)  # Note: typo in original code
            mc_returns_list = []
            
            for traj_rewards in rewards[:self.max_trajectories_to_log]:
                returns = []
                G = 0
                # Compute returns backwards
                for r in reversed(traj_rewards):
                    G = r + gamma * G
                    returns.append(G)
                mc_returns_list.append(torch.tensor(list(reversed(returns))))
            
            if mc_returns_list:
                return torch.cat(mc_returns_list)
            return None
        except:
            return None

    def _get_q_value_predictions(self, data):
        """Get Q-value predictions from the critic for comparison"""
        try:
            # Process a small batch to get Q-values
            batch_tokens = batch_traj_process(
                data['task_description'][:self.max_trajectories_to_log],
                data['obs'][:self.max_trajectories_to_log], 
                data['subtask'][:self.max_trajectories_to_log],
                self.engine.tokenizer
            ).to(self.engine.device)
            
            with torch.no_grad():
                hidden_states, _, action_end_mask = self.engine.get_hidden_states(batch_tokens)
                q_values = self.engine.critic_forward(hidden_states)
                valid_q_values, _ = self.extract_valid(q_values, action_end_mask)
                
            return valid_q_values.flatten()
        except:
            return None

    def log_trajectories_to_wandb(self, batch, step):
        """Log AWAC trajectories to WandB with reward/value information"""
        if self.engine.local_rank != 0:  # Only log on main process
            return
        
        try:
            # Extract data for all levels (expert, medium, low)
            expert_data = batch['high']
            medium_data = batch['medium'] 
            low_data = batch['low']
            
            # 1. AWAC-specific Timeline with Q-values and Rewards
            timeline_html = self._create_awac_timeline(expert_data, medium_data, low_data, step)
            wandb.log({"awac_trajectories/timeline": wandb.Html(timeline_html)}, step=step)
            
            # 2. Reward/Value Analysis Dashboard
            self._log_reward_analysis(expert_data, medium_data, step)
            
            # 3. Policy Performance Comparison Table
            self._log_policy_comparison(expert_data, medium_data, low_data, step)
            
            # 4. Q-value vs Monte Carlo Analysis (NEW!)
            self._log_qvalue_analysis(expert_data, medium_data, step)
            
        except Exception as e:
            print(f"Warning: Failed to log AWAC trajectories to WandB: {e}")
            wandb.log({"awac_trajectory_error": str(e)}, step=step)

    def _create_awac_timeline(self, expert_data, medium_data, low_data, step):
        """Create AWAC-specific timeline with rewards and Q-values"""
        html = f"""
        <html>
        <head>
            <style>
                .awac-timeline {{ font-family: Arial, sans-serif; margin: 20px; }}
                .level-section {{ border: 2px solid #333; margin: 20px 0; padding: 15px; border-radius: 10px; }}
                .expert {{ background: linear-gradient(90deg, #e8f5e8, #d4f1d4); border-color: #4caf50; }}
                .medium {{ background: linear-gradient(90deg, #fff3e0, #ffe0b2); border-color: #ff9800; }}
                .low {{ background: linear-gradient(90deg, #e3f2fd, #bbdefb); border-color: #2196f3; }}
                .level-header {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
                .trajectory {{ margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.7); border-radius: 5px; }}
                .step {{ margin: 5px 0; padding: 8px; background: #f9f9f9; border-radius: 3px; border-left: 3px solid #ddd; }}
                .reward {{ color: #d32f2f; font-weight: bold; }}
                .task {{ color: #1976d2; font-weight: bold; }}
                .action {{ color: #7b1fa2; }}
                .obs {{ color: #388e3c; font-style: italic; }}
                .metrics {{ background: #f5f5f5; padding: 5px; margin: 5px 0; border-radius: 3px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="awac-timeline">
                <h2>AWAC Hierarchical Training - Step {step}</h2>
        """
        
        # Expert trajectories
        html += self._create_level_section("EXPERT", expert_data, "expert")
        # Medium trajectories  
        html += self._create_level_section("MEDIUM", medium_data, "medium")
        # Low-level trajectories
        html += self._create_level_section("LOW-LEVEL", low_data, "low")
        
        html += "</div></body></html>"
        return html

    def _create_level_section(self, level_name, data, css_class):
        """Create a section for one level of the hierarchy"""
        html = f"""
            <div class="level-section {css_class}">
                <div class="level-header">{level_name} POLICY</div>
        """
        
        # Extract data (limit to first few trajectories)
        max_trajs = min(2, len(data.get('task_description', [])))
        
        for i in range(max_trajs):
            task_desc = data['task_description'][i] if 'task_description' in data else data.get('subtask', ['N/A'])[i]
            obs_list = data['obs'][i] if 'obs' in data else []
            action_list = data.get('subtask', data.get('action', []))[i] if i < len(data.get('subtask', data.get('action', []))) else []
            rewards = data.get('reward', [[0]*len(action_list)])[i] if 'reward' in data else [0]*len(action_list)
            dones = data.get('done', [[0]*len(action_list)])[i] if 'done' in data else [0]*len(action_list)
            
            html += f"""
                <div class="trajectory">
                    <div class="task">Task: {str(task_desc)[:150]}...</div>
                    <div class="metrics">
                        Trajectory Length: {len(action_list)} | 
                        Total Reward: {sum(rewards):.2f} | 
                        Completed: {"Yes" if any(dones) else "No"}
                    </div>
            """
            
            # Show first few steps
            for j, (obs, action, reward) in enumerate(zip(obs_list[:5], action_list[:5], rewards[:5])):
                html += f"""
                    <div class="step">
                        <div class="obs">Obs: {str(obs)[:100]}...</div>
                        <div class="action">Action: {str(action)[:80]}...</div>
                        <div class="reward">Reward: {reward:.3f}</div>
                    </div>
                """
            
            if len(action_list) > 5:
                html += f'<div class="step">... and {len(action_list)-5} more steps</div>'
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _log_reward_analysis(self, expert_data, medium_data, step):
        """Log reward and performance analysis"""
        expert_rewards = expert_data.get('reward', [])
        medium_rewards = medium_data.get('reward', [])
        
        if expert_rewards and medium_rewards:
            # Calculate statistics
            expert_total_rewards = [sum(traj_rewards) for traj_rewards in expert_rewards]
            medium_total_rewards = [sum(traj_rewards) for traj_rewards in medium_rewards]
            
            expert_avg_reward = sum(expert_total_rewards) / len(expert_total_rewards)
            medium_avg_reward = sum(medium_total_rewards) / len(medium_total_rewards)
            
            wandb.log({
                "awac_stats/expert_avg_reward": expert_avg_reward,
                "awac_stats/medium_avg_reward": medium_avg_reward,
                "awac_stats/reward_gap": expert_avg_reward - medium_avg_reward,
                "awac_stats/expert_max_reward": max(expert_total_rewards),
                "awac_stats/medium_max_reward": max(medium_total_rewards),
                "awac_stats/expert_traj_length": sum(len(r) for r in expert_rewards) / len(expert_rewards),
                "awac_stats/medium_traj_length": sum(len(r) for r in medium_rewards) / len(medium_rewards),
            }, step=step)

    def _log_policy_comparison(self, expert_data, medium_data, low_data, step):
        """Create comparison table across policy levels"""
        comparison_columns = ["Policy_Level", "Avg_Reward", "Avg_Length", "Success_Rate", "Sample_Task"]
        comparison_data = []
        
        for level_name, data in [("Expert", expert_data), ("Medium", medium_data), ("Low", low_data)]:
            if 'reward' in data and data['reward']:
                rewards = data['reward']
                avg_reward = sum(sum(r) for r in rewards) / len(rewards)
                avg_length = sum(len(r) for r in rewards) / len(rewards)
                success_rate = sum(any(data.get('done', [[]])[i]) for i in range(len(rewards))) / len(rewards)
                sample_task = str(data.get('task_description', data.get('subtask', ['N/A']))[0])[:100]
                
                comparison_data.append([
                    level_name,
                    f"{avg_reward:.3f}",
                    f"{avg_length:.1f}",
                    f"{success_rate:.2%}",
                    sample_task + "..."
                ])
        
        if comparison_data:
            comparison_table = wandb.Table(columns=comparison_columns, data=comparison_data)
            wandb.log({"awac_trajectories/policy_comparison": comparison_table}, step=step)

    def _log_qvalue_stats(self, expert_data, medium_data, step):
        """Log Q-value related statistics (placeholder for now)"""
        # This would require computing Q-values for the current batch
        # For now, log basic trajectory statistics
        stats = {
            "awac_stats/expert_batch_size": len(expert_data.get('reward', [])),
            "awac_stats/medium_batch_size": len(medium_data.get('reward', [])),
            "awac_stats/total_trajectories": len(expert_data.get('reward', [])) + len(medium_data.get('reward', [])),
        }
        wandb.log(stats, step=step)

    def get_policy_q(self, batch_prompt, batch_obs_list, batch_action_list):
        """
        Args:
            batch_obs_list: List[List[str]], shape: (batch, steps+1)
            batch_action_list: List[List[int]], shape: (batch, steps)
        Returns:
            q_values: Q(s, a~π), (batch, max_steps)
        """
        q_values = []
        for prompt, obs_list, action_list in zip(batch_prompt, batch_obs_list, batch_action_list):
            obs_list = obs_list[:-1]
            traj_len = len(obs_list)
            q_list = []
            traj_token = self.engine.tokenizer(prompt, return_tensors='pt')
            for t in range(traj_len):
                obs_token = self.engine.tokenizer(obs_list[t], return_tensors='pt')
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], obs_token["input_ids"]], dim = 1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                
                pi_action = self.engine.generate_action(copy.deepcopy(traj_token))[0]
                pi_action_token = self.engine.tokenizer(pi_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                input_token = {"input_ids": torch.cat([traj_token["input_ids"], pi_action_token["input_ids"]], dim=1).to(self.engine.device),
                               "attention_mask": torch.cat([traj_token["attention_mask"], pi_action_token["attention_mask"]], dim=1).to(self.engine.device)}
                hidden_states = self.engine.base(**input_token, output_hidden_states=True).hidden_states[-1][:,-1] # (1, hidden_dim)
                q_value, _ = self.engine.target_critic(hidden_states)
                q_list.append(q_value)

                action_token = self.engine.tokenizer(action_list[t]+self.engine.tokenizer.eos_token, return_tensors='pt')
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], action_token["input_ids"]], dim = 1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], action_token["attention_mask"]], dim = 1)

            q_values.append(torch.cat(q_list))
        q_values = pad_sequence(q_values, batch_first=True, padding_value=0.0)
        return q_values
                
                
    def extract_valid(self, value, valid_mark):
        """
        Args:
            value: (batch, seq_len)
            valid_mark: (batch, seq_len), where 1 indicates extraction position
        Returns:
            valid_value: The extracted sequence -> (batch, padding_steps)
            mask: padding position->(batch, padding_steps)(1111...00)
        """
        batch_size = value.size(0)
        max_valid_len = valid_mark.sum(dim=1).max().item()

        valid_value = torch.zeros(batch_size, max_valid_len, device=value.device)
        mask = torch.zeros(batch_size, max_valid_len, device=value.device)
        for i in range(batch_size):
            valid_idx = torch.where(valid_mark[i] == 1)[0]
            valid_len = valid_idx.size(0) # same to value and q_value
            
            valid_value[i, :valid_len] = value[i][valid_idx]
            mask[i, :valid_len] = 1

        return valid_value, mask
    

    def prepare_tensor(self, rewards, dones):
        """
        Args:
            rewards: List[(batch, steps)]
            dones: List[(batch, steps)]
        Returns:
            reward: tensor -> (batch, padding_steps)
            dones: tensor -> (batch, padding_steps)
        """
        # process reward
        reward_list = [torch.tensor(seq, dtype=torch.float, device=self.engine.device) for seq in rewards]
        done_list = [torch.tensor(seq, dtype=torch.float, device=self.engine.device) for seq in dones]
        
        # padding
        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor = pad_sequence(done_list, batch_first=True, padding_value=0)
        
        return reward_tensor, done_tensor

    