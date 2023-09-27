import IPython
if IPython.get_ipython() is not None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
import gym, torch, tqdm, random, os
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Tuple, Iterable, Dict, Union, Optional
from models import BaseDensePolicyNet, BaseSparsePolicyNet

os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["ENV_SEED"] = "3407"

class ReplayMemory:
    def __init__(self, memory_capacity:int=10_000):
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=memory_capacity)

    def updateMemory(self, observation, action, reward, future_observation, terminated):
        self.memory.append([
            observation,
            action,
            reward,
            future_observation,
            terminated,
        ])
        return self

    def sampleBatch(self, n:int, device:str):
        sample = random.sample(self.memory, k=n)

        observations = torch.stack([i[0] for i in sample], dim=0).to(device)
        actions = torch.stack([i[1] for i in sample], dim=0).to(device)
        rewards = torch.stack([i[2] for i in sample], dim=0).to(device)
        future_observations = torch.stack([i[3] for i in sample], dim=0).to(device)
        terminated_states = torch.BoolTensor([i[4] for i in sample]).to(device)

        return observations, actions, rewards, future_observations, terminated_states

    def __repr__(self):
        return f'{self.memory}'

    def clear(self):
        self.memory = deque(maxlen=self.memory_capacity)
        return self


def getActionQValues(
        agent_model: Union[BaseDensePolicyNet, BaseSparsePolicyNet], 
        observation:torch.Tensor):
    observation = observation.to(os.getenv("DEVICE"))
    with torch.no_grad():
        if issubclass(agent_model.__class__, BaseSparsePolicyNet):
            q_values, _ = agent_model(observation)
        else:
            q_values = agent_model(observation)
    return q_values


def getActionFromQValues(q_values: torch.Tensor, epsilon: float):
    if epsilon < np.random.uniform(0, 1):
        return torch.argmax(q_values, dim=-1).cpu()
    return torch.randint(0, q_values.shape[-1], size=q_values.shape)[:, 0]


def epsilonDecayWithCosineAnnealing(
    min_epsilon:float, 
    max_epsilon:float, 
    current_episode:int, 
    max_episode:int) -> float:

    cosine_function = np.cos((current_episode / max_episode) * np.pi)
    epsilon = min_epsilon + (0.5*(max_epsilon - min_epsilon) * (1 + cosine_function))
    return epsilon


def optimimumQValue(
    future_q_values: torch.Tensor, 
    rewards: torch.Tensor,
    gamma: float, 
    terminated_states: torch.BoolTensor) -> torch.Tensor:

    max_q_values = torch.max(future_q_values, dim=-1).values
    rewards = rewards.reshape(-1)

    optimum_values = rewards + (gamma*max_q_values*(1-terminated_states.type(torch.int64)))
    return optimum_values


def singleTrainAgentStep(
    agent_model: Union[BaseDensePolicyNet, BaseSparsePolicyNet],
    target_model: Union[BaseDensePolicyNet, BaseSparsePolicyNet],
    lossfunc: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_memory: ReplayMemory, 
    gamma: float, 
    batch_size: int):

    optimizer.zero_grad()
    # sample from replay memory
    (
        current_observations, 
        actions, 
        rewards, 
        future_observations, 
        terminated_states
    ) = replay_memory.sampleBatch(n=batch_size, device=os.getenv("DEVICE"))

    # get agent and target Q values from networks
    if issubclass(agent_model.__class__, BaseSparsePolicyNet):
        agent_Qs, agent_sparsification_penalty = agent_model(current_observations)
        future_Qs, _ = target_model(future_observations)
    else:
        agent_Qs = agent_model(current_observations)
        future_Qs = target_model(future_observations)
    
    optimal_Q = optimimumQValue(future_Qs, rewards, gamma, terminated_states)
    agent_Q = agent_Qs.gather(-1, actions.type(torch. int64)).reshape(-1)
    
    if issubclass(agent_model.__class__, BaseSparsePolicyNet):
        loss: torch.Tensor = lossfunc((agent_Q, agent_sparsification_penalty), optimal_Q)
    else:
        loss: torch.Tensor = lossfunc(agent_Q, optimal_Q)
    loss.backward()

    for param in agent_model.parameters():
        param.grad.data.clamp(-1.0, 1.0)
    optimizer.step()
    return loss.item()


def saveModelParams(params, filename:str, dir:str="../params"):
    if not os.path.isdir(dir):os.mkdir(dir)
    torch.save(params, os.path.join(dir, filename))


def train(
        env: gym.Env, 
        agent_model: Union[BaseDensePolicyNet, BaseSparsePolicyNet],
        target_model: Union[BaseDensePolicyNet, BaseSparsePolicyNet],
        replay_memory: ReplayMemory,
        lossfunc: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        episodes: int=10_000,
        batch_size: int=64,
        gamma: float=0.99,
        min_epsilon: float=0.0,
        max_epsilon: float=1.0,
        max_episode_reward: Union[int, float]=200,
        max_episode_steps: Optional[int]=None,
        continue_after_max_reward: bool=False,
        target_network_update_step: int=5,
        weights_filename: Optional[str]="agent_params.pth.tar",
        save_best_weights: bool=True,
        verbose: bool=False,
        close_env: bool=False,
        scale_obs: bool=False,
        obs_low: Optional[Union[int, np.ndarray, torch.Tensor]]=None,
        obs_high: Optional[Union[int, np.ndarray, torch.Tensor]]=None,
        ) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:
    
    train_performance = {
        'rewards':[],
    }

    best_rewards = {
        'episodes': [],
        'rewards': []
    }

    best_episode_reward = -np.inf
    epsilon = max_epsilon

    max_episode_steps = max_episode_steps or env._max_episode_steps

    agent_model.train()

    if scale_obs:
        obs_low = obs_low or env.observation_space.low
        obs_high = obs_high or env.observation_space.high

        obs_low = obs_low if not isinstance(obs_low, torch.Tensor) else obs_low.numpy()
        obs_high = obs_high if not isinstance(obs_high, torch.Tensor) else obs_high.numpy()
    else:
        obs_low = 0
        obs_high = 1

    for episode in tqdm.tqdm(range(0, episodes)):
        observation, _ = env.reset(seed=int(os.environ["ENV_SEED"]))
        termination_state = False
        episode_reward = 0
        
        for _ in range(0, max_episode_steps):
            observation = (observation - obs_low) / (obs_high - obs_low)
            _observation = torch.Tensor(observation).unsqueeze(dim=0)

            action = getActionFromQValues(getActionQValues(agent_model, _observation), epsilon)
            future_observation, reward, terminated, truncated, _ = env.step(action.item())
            future_observation = (future_observation - obs_low) / (obs_high - obs_low)
            
            # set termination_state to True or False depending on these values
            termination_state = any([terminated, truncated])
            episode_reward += reward

            # store experience in replay memory
            replay_memory.updateMemory(
                torch.tensor(observation, dtype=_observation.dtype), 
                torch.tensor(action), 
                torch.tensor(reward, dtype=_observation.dtype), 
                torch.tensor(future_observation, dtype=_observation.dtype), 
                torch.tensor(termination_state)
            )

            if len(replay_memory.memory) >= batch_size:

                singleTrainAgentStep(
                    agent_model,
                    target_model,
                    lossfunc,
                    optimizer,
                    replay_memory, 
                    gamma, 
                    batch_size, 
                )

            if termination_state: break

            observation = future_observation
        
        # track rewards for each episode
        train_performance['rewards'].append(episode_reward)

        # epsilon_decay
        epsilon = epsilonDecayWithCosineAnnealing(min_epsilon, max_epsilon, episode+1, episodes)

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            if verbose:
                print(f'best total reward for episode {episode+1}: {episode_reward}')
            best_rewards["episodes"].append(episode)
            best_rewards['rewards'].append(best_episode_reward)

            if save_best_weights:
                saveModelParams(agent_model.state_dict(), weights_filename)

        # update target network
        if episode == 0 or episode % target_network_update_step == 0:
            target_model.load_state_dict(agent_model.state_dict())

        # break if maximum reward is reached
        if not continue_after_max_reward:
            if best_episode_reward >= max_episode_reward: break

    if close_env: env.close()

    return train_performance, best_rewards