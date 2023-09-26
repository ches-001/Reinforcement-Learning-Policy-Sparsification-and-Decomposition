import gym
import numpy as np
from typing import Union, Tuple, Dict, Any
from collections import deque

class FormatFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, scale: bool=False):

        super(FormatFrameWrapper, self).__init__(env)

        self._scale = scale
        obs, _ = self.reset()

        self.observation_space = gym.spaces.Box(
            low=0.0 if self._scale else 0, 
            high=1.0 if self._scale else 255, 
            shape=obs.shape, 
            dtype=np.uint8 if not self._scale else np.float32
        )

    def observation(self, obs: Union[np.ndarray, gym.wrappers.frame_stack.LazyFrames])->np.ndarray:
        if isinstance(obs, gym.wrappers.frame_stack.LazyFrames):
            obs = np.array(obs)

        obs = np.array(obs)
        if self._scale: 
            obs = obs.astype(np.float32) / 255
        
        if obs.ndim==3:
            if obs.shape[-1] == 1:
                obs = np.squeeze(obs, axis=-1)
            else:
                obs = obs.transpose(2, 0, 1)

        if obs.ndim==4:
            if obs.shape[-1] == 1:
                obs = np.squeeze(obs, axis=-1)
            else:
                obs = obs.transpose(0, 3, 1, 2)

        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, n_skip: int):
        super(MaxAndSkipEnv, self).__init__(env)

        self._observation_buffer = deque(maxlen=2)
        self._n_skip = n_skip

    def step(self, action: int)->Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        true_info = {}
        terminated = None
        truncated = None

        for i in range(self._n_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            true_info.update(info)
            self._observation_buffer.append(observation)

            if terminated or truncated: break
        
        max_observation_frame = np.max(
                np.stack(self._observation_buffer, axis=0),axis=0)

        return (
            max_observation_frame,
            total_reward,
            terminated,
            truncated,
            true_info
        )

    def reset(self, **kwargs)->Tuple[np.ndarray, Dict[str, Any]]:
        self._observation_buffer.clear()
        observation, info = self.env.reset(**kwargs)
        self._observation_buffer.append(observation)

        return observation, info

