import os
import deepspeed
import torch
from transformers import AutoTokenizer
from util.model import Policy
from util.replay_buffer import HierarchyDataset, batch_traj_process
from alg.bc import Agent
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from prompt.inst import high_prompt, low_prompt

class GLIDER:
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=[{"params": [p for p in hierarcy_policy.base.parameters() if p.requires_grad], 
                                                                        "lr": args["lr"]}],
                                                     config=args["ds_config"])
        
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

    def update_policy(self):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        for epoch in range(self.args['epochs']):
            high_epoch_loss, low_epoch_loss = 0.0, 0.0
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,] or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],      or "obs": [group_nums, steps+1]
                        "subtask":[traj_nums, steps],    or "action": [group_nums, steps]
                        ...
                        }
                    pi(a|s)/action_token_len   
                """
                
                # Log trajectories to WandB every N steps (ADD THIS CALL)
                if (self.engine.local_rank == 0 and 
                    self.global_step.item() % self.log_interval == 0):
                    self.log_trajectories_to_wandb(batch, self.global_step.item())
                
                # high level
                batch_high_tokens = batch_traj_process(batch['high']['task_description'],
                                                       batch['high']['obs'],
                                                       batch['high']['subtask'],
                                                       self.engine.tokenizer).to(self.engine.device)

                high_log_probs, high_masks = self.engine.get_log_prob(batch_high_tokens)
                high_valid_log_prob = Agent.extract_valid_action_probs(self, high_log_probs, high_masks,
                                                                       max(batch_high_tokens['action_end_mask'].sum(dim=1)))
                high_loss = -high_valid_log_prob.mean()

                # low level
                batch_low_tokens = batch_traj_process(batch['low']['subtask'],
                                                      batch['low']['obs'],
                                                      batch['low']['action'],
                                                      self.engine.tokenizer).to(self.engine.device)
                low_log_probs, low_masks = self.engine.get_log_prob(batch_low_tokens)
                low_valid_log_prob = Agent.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                       max(batch_low_tokens['action_end_mask'].sum(dim=1)))
                low_loss = -low_valid_log_prob.mean()

                self.engine.backward(high_loss+low_loss)
                self.engine.step()

                high_epoch_loss += high_loss.item()
                low_epoch_loss += low_loss.item()
                self.global_step += 1

                if self.engine.local_rank == 0:  # Only log on the main process
                    print(f"train; step:{self.global_step.item()}; high:{high_loss.item()}; low:{low_loss.item()}; loss:{(low_loss+high_loss).item()}")
                    wandb.log({
                        'step_loss/high': high_loss.item(),
                        'step_loss/low': low_loss.item(),
                        'step_loss/bc': (high_loss+low_loss).item()
                    }, step=self.global_step.item())
            
            if self.engine.local_rank == 0: # Only log on the main process
                print(f"hierarcy-train; epoch{epoch}, high:{high_epoch_loss}, low:{low_epoch_loss}")
                Agent.save_policy(self)
                wandb.log({
                    'epoch_loss/high': high_epoch_loss,
                    'epoch_loss/low': low_epoch_loss,
                    'epoch_loss/bc': high_epoch_loss+low_epoch_loss
                }, step=epoch)

    def log_trajectories_to_wandb(self, batch, step):
        """Log sample trajectories to WandB using multiple visualization methods"""
        if self.engine.local_rank != 0:  # Only log on main process
            return
        
        try:
            task_descriptions = batch['high']['task_description'][:self.max_trajectories_to_log]
            high_observations = batch['high']['obs'][:self.max_trajectories_to_log]
            high_subtasks = batch['high']['subtask'][:self.max_trajectories_to_log]
            low_subtasks = batch['low']['subtask'][:self.max_trajectories_to_log]
            low_observations = batch['low']['obs'][:self.max_trajectories_to_log]
            low_actions = batch['low']['action'][:self.max_trajectories_to_log]
            
            # 1. Trajectory Timeline Visualization (HTML)
            timeline_html = self._create_trajectory_timeline(
                task_descriptions, high_observations, high_subtasks, 
                low_observations, low_actions, step
            )
            wandb.log({"trajectories/timeline": wandb.Html(timeline_html)}, step=step)
            
            # 2. Hierarchical Flow Chart
            flow_chart = self._create_hierarchy_flowchart(
                task_descriptions, high_subtasks, low_actions, step
            )
            wandb.log({"trajectories/hierarchy_flow": wandb.Html(flow_chart)}, step=step)
            
            # 3. Interactive Trajectory Table (Improved)
            self._log_interactive_tables(
                task_descriptions, high_observations, high_subtasks,
                low_subtasks, low_observations, low_actions, step
            )
            
            # 4. Trajectory Statistics Dashboard
            self._log_trajectory_stats(
                high_observations, high_subtasks, low_observations, low_actions, step
            )
            
            # 5. Text-based Trajectory Narratives
            self._log_trajectory_narratives(
                task_descriptions, high_subtasks, low_actions, step
            )
            
        except Exception as e:
            print(f"Warning: Failed to log trajectories to WandB: {e}")
            wandb.log({"trajectory_error": str(e)}, step=step)

    def _create_trajectory_timeline(self, task_descriptions, high_obs, high_subtasks, low_obs, low_actions, step):
        """Create an HTML timeline visualization"""
        html = f"""
        <html>
        <head>
            <style>
                .timeline {{ font-family: Arial, sans-serif; margin: 20px; }}
                .trajectory {{ border: 2px solid #3498db; margin: 20px 0; padding: 15px; border-radius: 10px; }}
                .task-desc {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; }}
                .level {{ margin: 10px 0; }}
                .high-level {{ background: #fff3e0; padding: 10px; border-left: 4px solid #ff9800; }}
                .low-level {{ background: #f3e5f5; padding: 10px; border-left: 4px solid #9c27b0; margin-left: 20px; }}
                .step {{ margin: 5px 0; padding: 5px; background: #f5f5f5; border-radius: 3px; }}
                .observation {{ color: #2e7d32; font-style: italic; }}
                .action {{ color: #c62828; font-weight: bold; }}
                .subtask {{ color: #f57c00; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="timeline">
                <h2>Hierarchical Trajectory Timeline - Step {step}</h2>
        """
        
        for i, (task_desc, h_obs, h_subtasks, l_obs, l_actions) in enumerate(zip(
            task_descriptions, high_obs, high_subtasks, low_obs, low_actions
        )):
            html += f"""
                <div class="trajectory">
                    <div class="task-desc">🎯 Task {i+1}: {str(task_desc)[:200]}...</div>
                    
                    <div class="high-level">
                        <h3>🔴 High-Level Policy</h3>
                        <div class="step"><span class="observation">Initial Obs:</span> {str(h_obs[0])[:150] if h_obs else 'None'}...</div>
            """
            
            if isinstance(h_subtasks, list):
                for j, subtask in enumerate(h_subtasks[:5]):  # Limit to 5 subtasks
                    html += f'<div class="step"><span class="subtask">Subtask {j+1}:</span> {str(subtask)[:100]}...</div>'
            
            html += """
                    </div>
                    
                    <div class="low-level">
                        <h3>🟢 Low-Level Policy</h3>
            """
            
            if isinstance(l_actions, list):
                for j, action in enumerate(l_actions[:8]):  # Limit to 8 actions
                    obs_text = str(l_obs[j])[:100] if l_obs and j < len(l_obs) else "No observation"
                    html += f"""
                        <div class="step">
                            <span class="observation">Obs {j+1}:</span> {obs_text}...<br>
                            <span class="action">Action {j+1}:</span> {str(action)[:80]}...
                        </div>
                    """
            
            html += "</div></div>"
        
        html += "</div></body></html>"
        return html

    def _create_hierarchy_flowchart(self, task_descriptions, high_subtasks, low_actions, step):
        """Create a flowchart showing task → subtasks → actions hierarchy"""
        html = f"""
        <html>
        <head>
            <style>
                .flowchart {{ font-family: Arial, sans-serif; margin: 20px; }}
                .flow-container {{ display: flex; flex-direction: column; align-items: center; }}
                .task-box {{ background: #2196f3; color: white; padding: 15px; margin: 10px; border-radius: 10px; text-align: center; max-width: 300px; }}
                .subtask-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px 0; }}
                .subtask-box {{ background: #ff9800; color: white; padding: 10px; border-radius: 5px; max-width: 200px; text-align: center; }}
                .action-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 5px; margin: 20px 0; }}
                .action-box {{ background: #4caf50; color: white; padding: 8px; border-radius: 3px; max-width: 150px; text-align: center; font-size: 12px; }}
                .arrow {{ font-size: 24px; color: #666; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="flowchart">
                <h2>🔄 Hierarchical Task Flow - Step {step}</h2>
        """
        
        for i, (task_desc, subtasks, actions) in enumerate(zip(task_descriptions, high_subtasks, low_actions)):
            html += f"""
                <div class="flow-container">
                    <div class="task-box">
                        <strong>Task {i+1}</strong><br>
                        {str(task_desc)[:150]}...
                    </div>
                    
                    <div class="arrow">⬇️</div>
                    
                    <div class="subtask-container">
            """
            
            if isinstance(subtasks, list):
                for subtask in subtasks[:4]:  # Limit subtasks
                    html += f'<div class="subtask-box">{str(subtask)[:80]}...</div>'
            
            html += """
                    </div>
                    
                    <div class="arrow">⬇️</div>
                    
                    <div class="action-container">
            """
            
            if isinstance(actions, list):
                for action in actions[:12]:  # Limit actions
                    html += f'<div class="action-box">{str(action)[:60]}...</div>'
            
            html += "</div></div><hr style='margin: 30px 0;'>"
        
        html += "</div></body></html>"
        return html

    def _log_interactive_tables(self, task_descriptions, high_obs, high_subtasks, low_subtasks, low_obs, low_actions, step):
        """Create enhanced interactive tables"""
        # Compact trajectory overview table
        overview_columns = ["Traj_ID", "Task_Preview", "High_Steps", "Low_Steps", "Success_Indicators"]
        overview_data = []
        
        for i, (task_desc, h_obs, h_subtasks, l_obs, l_actions) in enumerate(zip(
            task_descriptions, high_obs, high_subtasks, low_obs, low_actions
        )):
            # Simple success heuristic (you can customize this)
            success_indicators = "✅ Complete" if (len(l_actions) if isinstance(l_actions, list) else 0) > 5 else "⚠️ Incomplete"
            
            overview_data.append([
                f"Traj_{i+1}",
                str(task_desc)[:100] + "...",
                len(h_subtasks) if isinstance(h_subtasks, list) else 0,
                len(l_actions) if isinstance(l_actions, list) else 0,
                success_indicators
            ])
        
        overview_table = wandb.Table(columns=overview_columns, data=overview_data)
        wandb.log({"trajectories/overview": overview_table}, step=step)
        
        # Detailed action sequence table
        action_columns = ["Traj_ID", "Step", "Observation", "Action", "Action_Type"]
        action_data = []
        
        for i, (l_obs, l_actions) in enumerate(zip(low_obs, low_actions)):
            if isinstance(l_actions, list) and isinstance(l_obs, list):
                for j, (obs, action) in enumerate(zip(l_obs[:10], l_actions[:10])):  # Limit to 10 steps
                    action_type = self._classify_action(str(action))
                    action_data.append([
                        f"Traj_{i+1}",
                        j+1,
                        str(obs)[:150] + "...",
                        str(action)[:100] + "...",
                        action_type
                    ])
        
        action_table = wandb.Table(columns=action_columns, data=action_data)
        wandb.log({"trajectories/action_sequence": action_table}, step=step)

    def _classify_action(self, action_str):
        """Simple action classification for better visualization"""
        action_lower = action_str.lower()
        if any(word in action_lower for word in ['move', 'go', 'walk', 'run']):
            return "🚶 Movement"
        elif any(word in action_lower for word in ['take', 'pick', 'grab', 'get']):
            return "✋ Manipulation"
        elif any(word in action_lower for word in ['open', 'close', 'turn']):
            return "🔧 Interaction"
        elif any(word in action_lower for word in ['look', 'examine', 'focus']):
            return "👁️ Observation"
        else:
            return "❓ Other"

    def _log_trajectory_stats(self, high_obs, high_subtasks, low_obs, low_actions, step):
        """Log comprehensive trajectory statistics"""
        stats = {}
        
        # High-level stats
        high_lengths = [len(obs) if isinstance(obs, list) else 0 for obs in high_obs]
        subtask_lengths = [len(subtasks) if isinstance(subtasks, list) else 0 for subtasks in high_subtasks]
        
        # Low-level stats
        low_lengths = [len(obs) if isinstance(obs, list) else 0 for obs in low_obs]
        action_lengths = [len(actions) if isinstance(actions, list) else 0 for actions in low_actions]
        
        stats.update({
            "trajectory_stats/high_obs_mean": sum(high_lengths) / len(high_lengths) if high_lengths else 0,
            "trajectory_stats/high_obs_max": max(high_lengths) if high_lengths else 0,
            "trajectory_stats/high_obs_min": min(high_lengths) if high_lengths else 0,
            "trajectory_stats/subtasks_mean": sum(subtask_lengths) / len(subtask_lengths) if subtask_lengths else 0,
            "trajectory_stats/low_obs_mean": sum(low_lengths) / len(low_lengths) if low_lengths else 0,
            "trajectory_stats/actions_mean": sum(action_lengths) / len(action_lengths) if action_lengths else 0,
            "trajectory_stats/hierarchy_ratio": (sum(action_lengths) / sum(subtask_lengths)) if sum(subtask_lengths) > 0 else 0,
            "trajectory_stats/batch_size": len(high_obs)
        })
        
        wandb.log(stats, step=step)

    def _log_trajectory_narratives(self, task_descriptions, high_subtasks, low_actions, step):
        """Create human-readable trajectory narratives"""
        narratives = []
        
        for i, (task_desc, subtasks, actions) in enumerate(zip(task_descriptions, high_subtasks, low_actions)):
            narrative = f"**Trajectory {i+1}:**\n"
            narrative += f"🎯 **Goal:** {str(task_desc)[:200]}...\n\n"
            
            narrative += "📋 **High-Level Plan:**\n"
            if isinstance(subtasks, list):
                for j, subtask in enumerate(subtasks[:3]):
                    narrative += f"   {j+1}. {str(subtask)[:100]}...\n"
            
            narrative += "\n⚡ **Action Execution:**\n"
            if isinstance(actions, list):
                for j, action in enumerate(actions[:6]):
                    narrative += f"   Step {j+1}: {str(action)[:80]}...\n"
            
            narratives.append(narrative)
        
        # Log as text
        full_narrative = "\n\n" + "="*50 + "\n\n".join(narratives)
        wandb.log({"trajectories/narrative": wandb.Html(f"<pre>{full_narrative}</pre>")}, step=step)