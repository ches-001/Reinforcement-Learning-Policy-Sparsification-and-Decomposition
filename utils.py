import torch, gym, os
import torch.nn as nn
import  torch, copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import PolicyNet, SparsePolicyNet, L0Linear
from torch.distributions.categorical import Categorical
from typing import Tuple, Union, Dict, Optional, List


# model weight initialisation function
def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if torch.is_tensor(m.bias):
            m.bias.data.fill_(0.01)


def save_model_params(params, filename:str, dir:str="params"):
    if not os.path.isdir(dir):os.mkdir(dir)
    torch.save(params, os.path.join(dir, filename))


def compute_svd(
        tensor: torch.Tensor, 
        rank: int, 
        is_sparse: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if is_sparse:
        least_dim = min(tensor.shape)
        if least_dim < rank: 
            rank = least_dim
        u, s, v = torch.svd_lowrank(tensor, q=rank)
        return u, torch.diag(s), v.T

    else:
        u, s, v_t = torch.linalg.svd(tensor, full_matrices=False)
        u = u[:, :rank]
        s = torch.diag(s)[:rank, :rank]
        v_t = v_t[:rank, :]
        return u, s, v_t


def compile_sparse_model(model: SparsePolicyNet) -> PolicyNet:
    _model = PolicyNet(
        model.input_size, 
        model.output_size, 
        model.hidden_size).to(model.fc1._layer.weight.data.device)
    
    layers = list(i[0] for i in model.named_children() if isinstance(i[1], L0Linear))

    for layer in layers:
        z, _ = getattr(getattr(model, layer), "_compute_mask_and_penalty")()
        w = getattr(model, layer)._layer.weight.data.clone() * z
        setattr(getattr(getattr(_model, layer), "weight"), "data", w)

        b = None
        if hasattr(getattr(model, layer)._layer, "bias"):
            b = getattr(model, layer)._layer.bias.data.clone()
            setattr(getattr(getattr(_model, layer), "bias"), "data", b)
        else:
            setattr(getattr(_model, layer), "bias", b)

    return _model


def low_rank_approximate_policy(
        model: Union[PolicyNet, SparsePolicyNet], 
        rank: int,
        lra_w_decimals: int=6
    ) -> Tuple[PolicyNet, float]:
    
    model = copy.deepcopy(model)
    is_sparse = False
    if isinstance(model, SparsePolicyNet):
        model = compile_sparse_model(model)
        is_sparse = True

    layers = list(i[0] for i in model.named_children() if isinstance(i[1], nn.Linear))
    original_w_sizes = []
    lra_w_d_sizes = []

    compute_tensor_size = lambda tensor : (tensor.element_size() * tensor.nelement()) / (1024**2)

    for layer in layers:
        w = getattr(model, layer).weight.data.clone()
        u, s, vt = compute_svd(w, rank=rank, is_sparse=is_sparse)
        lra_w = torch.matmul(torch.matmul(u, s), vt)
        if is_sparse:
            lra_w = lra_w.round(decimals=lra_w_decimals)
        setattr(getattr(getattr(model, layer), "weight"), "data", lra_w)

        bias_size = 0
        if hasattr(getattr(model, layer), "bias"):
            bias_size = compute_tensor_size(getattr(model, layer).bias.data)

        original_w_sizes.append(compute_tensor_size(w) + bias_size)
        lra_w_d_sizes.append(compute_tensor_size(u) + compute_tensor_size(s) + compute_tensor_size(vt) + bias_size)

    policy_size = sum(original_w_sizes)
    decomposed_policy_size = sum(lra_w_d_sizes)
    percent_size_diff = (policy_size - decomposed_policy_size) / policy_size
    return model, percent_size_diff   


def compute_weight_sparsity(model: Union[PolicyNet, SparsePolicyNet]) -> Tuple[Dict, float]:
        if isinstance(model, SparsePolicyNet):
              return model.compute_weight_sparsity()
        
        layers = list(i[0] for i in model.named_children() if isinstance(i[1], L0Linear))
        get_layer_sparisty = lambda z : z[z==0].shape[0] / (z.shape[0] * z.shape[1])

        layer_sparsities = {}
        for layer in layers:
            layer_sparsities[layer] = get_layer_sparisty(getattr(model, layer).weight.data)

        percent_sparisty = sum(list(layer_sparsities.values())) / len(layer_sparsities)
        return layer_sparsities, percent_sparisty


def evaluate_policy(
        env: gym.Env, 
        policy_agent: Union[PolicyNet, SparsePolicyNet], 
        max_episode_steps: int=1000,
        close_env: bool=False,
        env_seed: Optional[int]=None
    ) -> Tuple[Union[int, float], float, float]:

    policy_agent.eval()
    obs, _ = env.reset(seed=env_seed)

    total_reward = 0
    avg_policy_entropy = 0
    pole_angles = []
    for _ in range(max_episode_steps):
        _obs = torch.tensor([obs], dtype=torch.float32, device=os.environ["DEVICE"])

        with torch.no_grad():
            if isinstance(policy_agent, SparsePolicyNet):
                action_logits, _ = policy_agent(_obs)
            else:
                action_logits = policy_agent(_obs)

        action_logits = action_logits.squeeze()
        action = action_logits.argmax()

        try:
            obs, reward, terminate, truncated, _ = env.step(action.item())
        except OverflowError:
            break

        policy_entropy = Categorical(logits=action_logits).entropy()
        avg_policy_entropy += policy_entropy.item()
        pole_angles.append(obs[2])

        total_reward += reward
        if truncated or terminate: break

    avg_policy_entropy /= max_episode_steps
    pole_angle_variance = (np.asarray(pole_angles).std())**2
    if close_env:
        env.close()
    return total_reward, avg_policy_entropy, pole_angle_variance


def run_evaluation(
        policy_agent: Union[PolicyNet, SparsePolicyNet], 
        render_mode: Optional[str]=None, 
        verbose: bool=True,
    ) -> Tuple[Union[int, float], float, float]:
    env = gym.make("CartPole-v1", render_mode=render_mode)

    eps_reward, avg_policy_entropy, pole_angle_variance = evaluate_policy(
        env, 
        policy_agent, 
        max_episode_steps=env._max_episode_steps, 
        close_env=True, 
        env_seed=int(os.environ["ENV_SEED"]),
    )
    if verbose:
        print(f"Evaluation reward of policy: {eps_reward}")
        print(f"Average policy entropy of policy: {round(avg_policy_entropy, 7)}")
        print(f"Pole angle variance of policy: {round(pole_angle_variance, 7)} rads")
    
    return eps_reward, avg_policy_entropy, pole_angle_variance


def plot_result_dataframe(
        df: pd.DataFrame, 
        x_axis: str, 
        figsize=(20, 21), 
        hm_figsize=(10, 7),
        plot_type: str="line", 
        with_heatmap: bool=False):
    assert plot_type in ["line", "bar"], "plot_type can only be line or bar"

    cols = list(df.columns.copy())
    cols.remove(x_axis)
    _, axs = plt.subplots(len(cols), 1, figsize=figsize)

    for idx, col in enumerate(cols):
        if plot_type =="line":
            axs[idx].grid(True)
            axs[idx].plot(df[x_axis], df[col])
        else:
            sns.barplot(data=df, x=x_axis, y=df[col], ax=axs[idx], color="blue")
        axs[idx].set_xlabel(x_axis)
        axs[idx].set_ylabel(col)
        axs[idx].set_title(f"{col} VS {x_axis}")
    plt.show()

    if with_heatmap:
        print("\n\n")
        plt.figure(figsize=hm_figsize)
        sns.heatmap(df.corr().fillna(0), annot=True, cmap="coolwarm", center=0)
        plt.title("Heatmap plot")
        plt.show()


def plot_train_performance(train_performance: Dict[str, List], best_rewards: Dict[str, List]):
    fig = plt.figure(figsize=(20, 5))
    episodes_idx = list(range(0, len(train_performance["rewards"])))
    plt.plot(episodes_idx, train_performance["rewards"])
    plt.scatter([i+1 for i in best_rewards["episodes"]], best_rewards["rewards"], label="best_rewards", c="orange")
    plt.legend()
    plt.title(f"Rewards plot for Dense Agent Policy (converged in {len(episodes_idx)} steps)")
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.show()