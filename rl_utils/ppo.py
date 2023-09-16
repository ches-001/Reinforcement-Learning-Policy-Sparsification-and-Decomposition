import IPython
if IPython.get_ipython() is not None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
import gym, torch, tqdm, os, copy
import numpy as np
from torch.distributions.categorical import Categorical
from typing import Tuple, Iterable, Dict, Union, Optional
from models import PolicyNet, SparsePolicyNet

os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

class PPOTrainer:
    def __init__(
            self, 
            policy_agent: Union[PolicyNet, SparsePolicyNet], 
            value_agent: Union[PolicyNet, SparsePolicyNet], 
            policy_optimizer: torch.optim.Optimizer,
            value_optimzier: torch.optim.Optimizer,
            *,
            epsilon: float=0.2, 
            kl_div_coef: float=0.01,
            n_policy_train_steps: int=100,
            n_value_train_steps: int=100,
            sparse_penalty_coef: Optional[float]=None,
            maximize_entropy: bool=False,
        ):
        
        self.policy_agent = policy_agent
        self.policy_agent.to(os.environ["DEVICE"] )
        self.value_agent = value_agent
        self.value_agent.to(os.environ["DEVICE"] )
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimzier
        self.epsilon = epsilon
        self.kl_div_coef = kl_div_coef
        self.n_policy_train_steps = n_policy_train_steps
        self.n_value_train_steps = n_value_train_steps
        self.sparse_penalty_coef = sparse_penalty_coef
        self.maximize_entropy = maximize_entropy

    
    def _custom_collate_fn(self, batch: Iterable[torch.Tensor]):
        print(type(batch), len(batch))

    def train_policy_net(
            self, 
            obs: torch.Tensor, 
            actions: torch.Tensor, 
            old_log_probs: torch.Tensor, 
            gaes: torch.Tensor):
        
        obs = obs.to(os.environ["DEVICE"] )
        actions = actions.to(os.environ["DEVICE"] )
        old_log_probs = old_log_probs.to(os.environ["DEVICE"] )
        gaes = gaes.to(os.environ["DEVICE"])


        for _ in range(self.n_policy_train_steps):
            self.policy_optimizer.zero_grad()

            if isinstance(self.policy_agent, SparsePolicyNet):
                new_logits, sparse_penalty = self.policy_agent(obs)
            
            else:
                new_logits = self.policy_agent(obs)

            action_dist = Categorical(logits=new_logits)
            new_log_probs = action_dist.log_prob(actions)
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_policy_ratio = (
                torch
                .exp(new_log_probs - old_log_probs)
                .clamp(1-self.epsilon, 1+self.epsilon)
            )
            policy_loss = -torch.min(policy_ratio * gaes, clipped_policy_ratio * gaes)

            if self.maximize_entropy:
                policy_loss = policy_loss - (self.kl_div_coef * action_dist.entropy())
            else:
                policy_loss = policy_loss + (self.kl_div_coef * action_dist.entropy())

            policy_loss = policy_loss.mean()
            if isinstance(self.policy_agent, SparsePolicyNet):
                policy_loss = policy_loss + (self.sparse_penalty_coef * sparse_penalty / obs.shape[0])
            policy_loss.backward()
            self.policy_optimizer.step()


    def train_value_net(self, obs: torch.Tensor, returns: torch.Tensor):
        obs = obs.to(os.environ["DEVICE"])
        returns = returns.to(os.environ["DEVICE"])

        for _ in range(self.n_value_train_steps):
            self.value_optimizer.zero_grad()

            if isinstance(self.value_agent, SparsePolicyNet):
                values, sparse_penalty = self.value_agent(obs)
                loss = ((values - returns) ** 2) + (self.sparse_penalty_coef * sparse_penalty / obs.shape[0])
            else:
                values = self.value_agent(obs)
                loss = (values - returns) ** 2
            loss = loss.mean()
            loss.backward()
            self.value_optimizer.step()


    def save_policy(self, filename:str, dir:str="params"):
        if not os.path.isdir(dir):os.mkdir(dir)
        torch.save(self.policy_agent.state_dict(), os.path.join(dir, filename))


def discount_reward(rewards: np.ndarray, gamma: float=0.99):
    # n_Σ_(k=t) (r_(t + k) * γ ^ (k))
    discounted_rewards = []
    discount_t = 0
    # Start from the last reward and work backward
    for r in reversed(rewards):
        discount_t = discount_t * gamma + r
        discounted_rewards.insert(0, discount_t)
    return discounted_rewards


def compute_gaes(rewards: np.ndarray, values: np.ndarray, gamma: float=0.99, lambda_: float=0.97):
    next_values = np.concatenate((values[1:], np.zeros(1)))
    deltas = rewards + (gamma * (next_values - values))
    gaes = []
    gae_t = 0
    # Start from the last reward and work backward
    for delta in reversed(deltas):
        gae_t = delta + (gamma * lambda_ * gae_t)
        gaes.insert(0, gae_t)
    return gaes


def rollout(
        env: gym.Env, 
        policy_agent: Union[PolicyNet, SparsePolicyNet], 
        value_agent: Union[PolicyNet, SparsePolicyNet], 
        max_steps: int=1000
    ):
    experience = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "act_log_probs": [],
    }

    obs, _ = env.reset()
    eps_reward = 0
    for _ in range(max_steps):
        _obs = torch.tensor([obs], dtype=torch.float32, device=os.environ["DEVICE"])

        if isinstance(policy_agent, SparsePolicyNet):
            action_logits, _ = policy_agent(_obs)
        else:
            action_logits = policy_agent(_obs)

        if isinstance(policy_agent, SparsePolicyNet):
            value, _ = value_agent(_obs)
        else:
            value = value_agent(_obs)

        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action).item()
        action = action.cpu().item()
        value = value.cpu().item()

        new_obs, reward, terminated, truncated, _ = env.step(action)

        experience["observations"].append(obs)
        experience["actions"].append(action)
        experience["rewards"].append(reward)
        experience["values"].append(value)
        experience["act_log_probs"].append(action_log_prob)

        eps_reward += reward
        obs = new_obs

        if truncated or terminated: break

    experience = {k:np.asarray(v) for k, v in experience.items()}
    return experience, eps_reward


def format_policy_data(data: Dict[str, np.ndarray]):
    data = copy.deepcopy(data)
    data["observations"] = torch.tensor(data["observations"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["actions"] = torch.tensor(data["actions"], dtype=torch.int32, device=os.environ["DEVICE"])
    data["rewards"] = torch.tensor(data["rewards"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["values"] = torch.tensor(data["values"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["returns"] = torch.tensor(data["returns"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["gaes"] = torch.tensor(data["gaes"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["act_log_probs"] = torch.tensor(data["act_log_probs"], dtype=torch.float32, device=os.environ["DEVICE"])
    return data


def train(
        env: gym.Env, 
        ppo: PPOTrainer, 
        *,
        n_episodes: int=100, 
        gamma: float=0.99, 
        lambda_: float=0.97, 
        max_rollout_steps: int=1000,
        max_episode_reward: Union[int, float]=200,
        continue_after_max_reward: bool=False,
        policy_weights_filename: Optional[str]="CartPolev0_PPO_policy_agent.pth.tar",
        save_best_weights: bool=True,
        verbose: bool=False,
        close_env: bool=False,
    ) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:

    train_performance = {
        'rewards':[],
    }

    best_rewards = {
        'episodes': [],
        'rewards': []
    }

    best_reward = -np.inf
    for episode in tqdm.tqdm(range(n_episodes)):
        data, episode_reward = rollout(
            env, 
            ppo.policy_agent, 
            ppo.value_agent, 
            max_rollout_steps
        )
        data["returns"] = discount_reward(data["rewards"], gamma)
        data["gaes"] = compute_gaes(data["rewards"], data["values"], gamma, lambda_)
        data = format_policy_data(data)

        ppo.train_policy_net(data["observations"], data["actions"], data["act_log_probs"], data["gaes"])
        ppo.train_value_net(data["observations"], data["returns"])

        train_performance["rewards"].append(episode_reward)

        if episode_reward >= best_reward:
            best_reward = episode_reward
            if verbose:
                print(f'best total reward for episode {episode+1}: {episode_reward}')
            best_rewards["episodes"].append(episode)
            best_rewards["rewards"].append(best_reward)

            if save_best_weights:
                ppo.save_policy(policy_weights_filename)

        if not continue_after_max_reward:
            if best_reward >= max_episode_reward: break

    if close_env:
        env.close()
    return train_performance, best_rewards