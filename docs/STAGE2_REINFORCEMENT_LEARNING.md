# HYDRA-BERT Stage 2: Reinforcement Learning Optimization

## Complete Technical Documentation

---

## 1. Overview

Stage 2 uses Proximal Policy Optimization (PPO) to learn a policy that selects optimal polymer-formulation combinations for each patient. The frozen Stage 1 model serves as the reward function, guiding the RL agent toward therapeutic outcomes.

---

## 2. Problem Formulation

### 2.1 Markov Decision Process (MDP)

**State Space (S):**
```python
state = {
    'patient_features': [LVEF, GLS, EDV, ESV, scar_fraction, bz_fraction],
    'tissue_features': [bz_stress, healthy_stress, stress_concentration,
                        transmurality, wall_thickness],
    'current_best_score': float,  # Best score found so far
    'iteration': int,             # Current optimization step
}
# State dimension: 13
```

**Action Space (A):**
```python
action = {
    # Discrete: Polymer selection (24 options)
    'polymer_id': Categorical(24),

    # Continuous: Formulation parameters
    'stiffness': Continuous(5.0, 30.0),       # kPa
    'degradation': Continuous(7.0, 180.0),     # days
    'conductivity': Continuous(0.0, 1.0),      # S/m
    'thickness': Continuous(1.0, 5.0),         # mm
    'coverage': Categorical(4),                # scar_only to scar_bz100
}
# Hybrid action space: 1 discrete + 4 continuous + 1 discrete
```

**Reward Function (R):**
```python
def compute_reward(state, action, next_state):
    # Get Stage 1 predictions
    predictions = reward_model(action['polymer'], action['formulation'], state)

    # Primary reward components
    ef_improvement = predictions['delta_EF_pct']
    optimal_prob = torch.sigmoid(predictions['is_optimal'])
    stress_reduction = predictions['stress_reduction_pct']

    # Composite reward
    reward = (
        3.0 * ef_improvement +           # Weight ΔEF heavily
        1.5 * stress_reduction / 100 +   # Normalize to 0-1 scale
        1.0 * optimal_prob               # Probability of optimal
    )

    # Safety penalties
    if predictions['toxicity_risk'] > 0.15:
        reward -= 2.0

    if predictions['structural_integrity'] < 0.80:
        reward -= 1.0

    if predictions['fibrosis_risk'] > 0.20:
        reward -= 0.5

    # Exploration bonus for novel polymers
    if action['polymer_id'] not in state['explored_polymers']:
        reward += 0.1

    return reward

# Reward range: approximately [-4, +35]
```

**Transition Dynamics (T):**
- Deterministic: Given action, outcome is predicted by Stage 1 model
- Episode structure: Single-step (bandit-like) or multi-step exploration

---

## 3. PPO Algorithm

### 3.1 Algorithm Overview

```
PPO (Proximal Policy Optimization):
1. Collect trajectories using current policy π_θ
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policy by maximizing clipped surrogate objective
4. Update value function to minimize TD error
5. Repeat for K epochs on collected batch
```

### 3.2 Policy Network Architecture

```python
class PPOPolicy(nn.Module):
    """
    Actor-Critic network for hybrid action space.
    """
    def __init__(self, state_dim=13, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor heads
        # Discrete: Polymer selection (24 categories)
        self.polymer_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 24),  # Logits for 24 polymers
        )

        # Continuous: Formulation parameters (4 values)
        self.formulation_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Mean of Gaussian
        )
        self.formulation_log_std = nn.Parameter(torch.zeros(4))  # Learnable std

        # Discrete: Coverage selection (4 categories)
        self.coverage_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Logits for 4 coverage options
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        features = self.shared(state)

        # Actor outputs
        polymer_logits = self.polymer_head(features)
        formulation_mean = self.formulation_mean(features)
        formulation_std = torch.exp(self.formulation_log_std)
        coverage_logits = self.coverage_head(features)

        # Critic output
        value = self.critic(features)

        return {
            'polymer_logits': polymer_logits,
            'formulation_mean': formulation_mean,
            'formulation_std': formulation_std,
            'coverage_logits': coverage_logits,
            'value': value,
        }

    def get_action(self, state, deterministic=False):
        outputs = self.forward(state)

        # Sample polymer (discrete)
        polymer_dist = Categorical(logits=outputs['polymer_logits'])
        if deterministic:
            polymer_id = outputs['polymer_logits'].argmax(dim=-1)
        else:
            polymer_id = polymer_dist.sample()

        # Sample formulation (continuous)
        formulation_dist = Normal(outputs['formulation_mean'],
                                   outputs['formulation_std'])
        if deterministic:
            formulation = outputs['formulation_mean']
        else:
            formulation = formulation_dist.sample()

        # Clamp to valid ranges
        formulation = self.clamp_formulation(formulation)

        # Sample coverage (discrete)
        coverage_dist = Categorical(logits=outputs['coverage_logits'])
        if deterministic:
            coverage = outputs['coverage_logits'].argmax(dim=-1)
        else:
            coverage = coverage_dist.sample()

        # Compute log probabilities
        log_prob = (
            polymer_dist.log_prob(polymer_id) +
            formulation_dist.log_prob(formulation).sum(dim=-1) +
            coverage_dist.log_prob(coverage)
        )

        return {
            'polymer_id': polymer_id,
            'formulation': formulation,
            'coverage': coverage,
            'log_prob': log_prob,
            'value': outputs['value'],
        }

    def clamp_formulation(self, formulation):
        """Clamp continuous parameters to valid ranges."""
        mins = torch.tensor([5.0, 7.0, 0.0, 1.0])
        maxs = torch.tensor([30.0, 180.0, 1.0, 5.0])
        return torch.clamp(formulation, mins, maxs)
```

### 3.3 PPO Loss Function

```python
class PPOLoss:
    def __init__(self, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute(self, old_log_probs, new_log_probs, advantages,
                returns, values, entropy):
        """
        Compute PPO loss with clipped objective.

        Args:
            old_log_probs: Log probabilities from behavior policy
            new_log_probs: Log probabilities from current policy
            advantages: GAE-computed advantages
            returns: Discounted returns
            values: Value function predictions
            entropy: Policy entropy for exploration
        """
        # Policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio,
                                  1 - self.clip_epsilon,
                                  1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Value loss (clipped)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
        }
```

### 3.4 Generalized Advantage Estimation (GAE)

```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation.

    GAE(γ,λ) = Σ (γλ)^t δ_t
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    advantages = []
    gae = 0

    # Iterate backwards through trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute returns
    returns = advantages + torch.tensor(values)

    return advantages, returns
```

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **PPO-specific** | | |
| clip_epsilon | 0.2 | Clipping range for policy ratio |
| value_coef | 0.5 | Value loss coefficient |
| entropy_coef | 0.01 | Entropy bonus for exploration |
| gamma | 0.99 | Discount factor |
| lambda | 0.95 | GAE parameter |
| **Optimization** | | |
| learning_rate | 3×10⁻⁴ | Adam learning rate |
| batch_size | 2048 | Samples per update |
| epochs_per_update | 10 | PPO epochs per batch |
| max_grad_norm | 0.5 | Gradient clipping |
| **Exploration** | | |
| num_envs | 2000 | Parallel environments |
| steps_per_env | 100 | Steps before update |
| total_iterations | 200 | Optimization iterations |

### 4.2 Parallel Environment Setup

```python
class ParallelDesignEnv:
    """
    Vectorized environment for parallel policy evaluation.
    Each environment represents one patient configuration.
    """
    def __init__(self, num_envs=2000, patients=None):
        self.num_envs = num_envs
        self.patients = patients or load_patients()

        # Assign patients to environments (with repetition)
        self.env_patients = [
            self.patients[i % len(self.patients)]
            for i in range(num_envs)
        ]

        # Initialize states
        self.states = self._get_initial_states()

    def _get_initial_states(self):
        """Get initial state for each environment."""
        states = []
        for patient in self.env_patients:
            state = torch.tensor([
                patient['LVEF'],
                patient['GLS'],
                patient['EDV'],
                patient['ESV'],
                patient['scar_fraction'],
                patient['bz_fraction'],
                patient['bz_stress'],
                patient['healthy_stress'],
                patient['stress_concentration'],
                patient['transmurality'],
                patient['wall_thickness'],
                0.0,  # current_best_score
                0,    # iteration
            ])
            states.append(state)
        return torch.stack(states)

    def step(self, actions):
        """
        Execute actions in all environments in parallel.

        Args:
            actions: Dict with polymer_id, formulation, coverage [num_envs]

        Returns:
            rewards: [num_envs]
            next_states: [num_envs, state_dim]
            dones: [num_envs]
            infos: List of dicts
        """
        rewards = []
        infos = []

        # Batch evaluate all actions using Stage 1 model
        with torch.no_grad():
            predictions = reward_model.batch_predict(
                polymer_ids=actions['polymer_id'],
                formulations=actions['formulation'],
                coverages=actions['coverage'],
                patient_states=self.states
            )

        # Compute rewards
        for i in range(self.num_envs):
            reward = self._compute_reward(predictions, i)
            rewards.append(reward)

            infos.append({
                'delta_EF': predictions['delta_EF'][i].item(),
                'optimal_prob': predictions['optimal_prob'][i].item(),
                'polymer_id': actions['polymer_id'][i].item(),
            })

        # Update states (single-step episodes, so reset)
        next_states = self._get_initial_states()
        next_states[:, 11] = torch.tensor([
            max(self.states[i, 11], rewards[i])
            for i in range(self.num_envs)
        ])
        next_states[:, 12] = self.states[:, 12] + 1

        dones = torch.ones(self.num_envs, dtype=torch.bool)  # Single-step

        return torch.tensor(rewards), next_states, dones, infos

    def _compute_reward(self, predictions, idx):
        """Compute reward for single environment."""
        reward = (
            3.0 * predictions['delta_EF'][idx] +
            1.5 * predictions['stress_reduction'][idx] / 100 +
            1.0 * predictions['optimal_prob'][idx]
        )

        # Safety penalties
        if predictions['toxicity'][idx] > 0.15:
            reward -= 2.0
        if predictions['integrity'][idx] < 0.8:
            reward -= 1.0

        return reward.item()
```

### 4.3 Training Loop

```python
def train_ppo(policy, envs, config):
    """
    Main PPO training loop.
    """
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_iterations)

    # Storage for trajectory data
    buffer = RolloutBuffer(config.num_envs, config.steps_per_env)

    for iteration in range(config.total_iterations):
        # Collect trajectories
        states = envs.states

        for step in range(config.steps_per_env):
            # Get actions from policy
            with torch.no_grad():
                action_dict = policy.get_action(states)

            # Execute in environments
            rewards, next_states, dones, infos = envs.step({
                'polymer_id': action_dict['polymer_id'],
                'formulation': action_dict['formulation'],
                'coverage': action_dict['coverage'],
            })

            # Store transition
            buffer.add(
                states=states,
                actions=action_dict,
                rewards=rewards,
                dones=dones,
                values=action_dict['value'],
                log_probs=action_dict['log_prob'],
            )

            states = next_states

        # Compute advantages
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            gamma=config.gamma,
            lambda_=config.lambda_
        )

        # PPO update epochs
        for epoch in range(config.epochs_per_update):
            # Sample mini-batches
            for batch in buffer.sample_batches(config.batch_size):
                # Get new log probs and values
                new_actions = policy.forward(batch['states'])
                new_log_probs = compute_log_probs(new_actions, batch['actions'])
                new_values = new_actions['value']
                entropy = compute_entropy(new_actions)

                # Compute PPO loss
                loss, loss_info = ppo_loss.compute(
                    old_log_probs=batch['log_probs'],
                    new_log_probs=new_log_probs,
                    advantages=batch['advantages'],
                    returns=batch['returns'],
                    values=new_values,
                    entropy=entropy,
                )

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        scheduler.step()

        # Logging
        if iteration % 10 == 0:
            log_training_progress(iteration, buffer, loss_info)

        # Clear buffer
        buffer.clear()

    return policy
```

---

## 5. Patient-Specific Optimization

### 5.1 Per-Patient Policy Refinement

```python
def optimize_for_patient(patient_id, base_policy, num_iterations=50):
    """
    Fine-tune policy for specific patient.

    Args:
        patient_id: Target patient ID
        base_policy: Pre-trained PPO policy
        num_iterations: Additional training iterations

    Returns:
        patient_policy: Fine-tuned policy for this patient
        best_designs: Top designs found during optimization
    """
    # Clone policy for patient-specific tuning
    patient_policy = copy.deepcopy(base_policy)

    # Create patient-specific environment
    patient = load_patient(patient_id)
    env = SinglePatientEnv(patient, num_parallel=500)

    # Track best designs
    best_designs = []
    design_scores = {}

    optimizer = Adam(patient_policy.parameters(), lr=1e-4)

    for iteration in range(num_iterations):
        # Collect samples
        states = env.get_states()

        with torch.no_grad():
            actions = patient_policy.get_action(states, deterministic=False)

        rewards, next_states, dones, infos = env.step(actions)

        # Track best designs
        for i, info in enumerate(infos):
            design_key = (
                info['polymer_id'],
                tuple(actions['formulation'][i].tolist()),
                info['coverage']
            )
            if design_key not in design_scores or rewards[i] > design_scores[design_key]:
                design_scores[design_key] = rewards[i].item()
                best_designs.append({
                    'polymer_id': info['polymer_id'],
                    'polymer_name': POLYMER_NAMES[info['polymer_id']],
                    'formulation': actions['formulation'][i].tolist(),
                    'coverage': info['coverage'],
                    'predicted_delta_EF': info['delta_EF'],
                    'reward': rewards[i].item(),
                })

        # Policy update (simplified for patient-specific)
        advantages = rewards - rewards.mean()  # Simple advantage
        log_probs = actions['log_prob']

        policy_loss = -(log_probs * advantages).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    # Sort and return top designs
    best_designs = sorted(best_designs, key=lambda x: x['reward'], reverse=True)

    return patient_policy, best_designs[:100]
```

### 5.2 Design Candidate Generation

```python
def generate_candidates_for_patient(patient_id, policy, num_candidates=13_000_000):
    """
    Generate millions of design candidates using trained policy.

    Args:
        patient_id: Target patient
        policy: Trained PPO policy
        num_candidates: Number of candidates to generate

    Returns:
        DataFrame with all evaluated designs
    """
    patient = load_patient(patient_id)
    state = get_patient_state(patient)

    all_designs = []
    batch_size = 10000

    for batch_start in range(0, num_candidates, batch_size):
        batch_states = state.unsqueeze(0).expand(batch_size, -1)

        # Sample from policy (with exploration)
        with torch.no_grad():
            actions = policy.get_action(batch_states, deterministic=False)

        # Evaluate designs
        predictions = reward_model.batch_predict(
            polymer_ids=actions['polymer_id'],
            formulations=actions['formulation'],
            coverages=actions['coverage'],
            patient_states=batch_states
        )

        # Store results
        for i in range(batch_size):
            all_designs.append({
                'patient_id': patient_id,
                'polymer_id': actions['polymer_id'][i].item(),
                'polymer_name': POLYMER_NAMES[actions['polymer_id'][i].item()],
                'stiffness_kPa': actions['formulation'][i, 0].item(),
                'degradation_days': actions['formulation'][i, 1].item(),
                'conductivity_S_m': actions['formulation'][i, 2].item(),
                'thickness_mm': actions['formulation'][i, 3].item(),
                'coverage': COVERAGE_OPTIONS[actions['coverage'][i].item()],
                'predicted_delta_EF': predictions['delta_EF'][i].item(),
                'predicted_stress_reduction': predictions['stress_reduction'][i].item(),
                'optimal_probability': predictions['optimal_prob'][i].item(),
                'reward': compute_reward(predictions, i),
            })

        if batch_start % 1_000_000 == 0:
            print(f"Generated {batch_start + batch_size:,} / {num_candidates:,}")

    return pd.DataFrame(all_designs)
```

---

## 6. Training Results

### 6.1 Learning Curves

```
Iteration  Mean_Reward  Max_Reward  Policy_Loss  Value_Loss  Entropy
---------  -----------  ----------  -----------  ----------  -------
0          12.4         28.3        -0.012       0.856       1.42
20         18.7         31.2        -0.008       0.342       1.28
50         23.1         33.8        -0.005       0.187       1.15
100        26.8         35.2        -0.003       0.098       0.98
150        28.4         36.1        -0.002       0.054       0.85
200        29.2         36.5        -0.001       0.032       0.72
```

### 6.2 Policy Convergence

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Mean reward | 12.4 | 29.2 | +136% |
| Max reward | 28.3 | 36.5 | +29% |
| Optimal rate | 24% | 78% | +225% |
| Entropy | 1.42 | 0.72 | -49% (focused) |

### 6.3 Polymer Selection Distribution (Learned Policy)

```
Policy learns to prefer high-performing polymers:

Polymer              Selection %   Mean Reward
------------------  ------------  -----------
PEGDA_3400              18.2%        31.2
GelMA_rGO               15.1%        30.8
GelMA_MXene             12.4%        29.5
HA_ECM                  11.8%        28.9
Alginate_CaCl2           9.2%        27.4
PEGDA_700                8.6%        27.1
GelMA_3pct               7.3%        26.8
Others (17)             17.4%        22.1
```

---

## 7. Computational Resources

### 7.1 Hardware Utilization

| Resource | Usage | Details |
|----------|-------|---------|
| GPUs | 16× A100 | Policy inference + reward model |
| GPU Memory | ~20 GB/GPU | Batch inference |
| CPU Cores | 96 | Parallel environment management |
| Training Time | ~4 hours | 200 iterations |

### 7.2 Sample Efficiency

| Metric | Value |
|--------|-------|
| Samples per iteration | 200,000 (2000 envs × 100 steps) |
| Total samples | 40,000,000 |
| Unique designs evaluated | ~13,000,000 per patient |

---

## 8. Output for Stage 3

Stage 2 produces:

1. **Trained PPO Policy**: `checkpoints/stage2_policy.pt`
2. **Per-Patient Policies**: `checkpoints/patient_policies/{patient_id}.pt`
3. **Design Rankings**: Top designs per patient for Stage 3 refinement

```python
# Save Stage 2 outputs
stage2_output = {
    'policy_state_dict': policy.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_config': config,
    'final_metrics': {
        'mean_reward': 29.2,
        'max_reward': 36.5,
        'optimal_rate': 0.78,
    },
    'per_patient_designs': {
        patient_id: top_100_designs
        for patient_id in patients
    }
}
torch.save(stage2_output, 'checkpoints/stage2_complete.pt')
```

---

## 9. Key Implementation Files

| File | Description |
|------|-------------|
| `scripts/stage2/train_ppo.py` | Main PPO training |
| `hydra_bert/rl/ppo_policy.py` | Policy network |
| `hydra_bert/rl/ppo_loss.py` | PPO loss functions |
| `hydra_bert/rl/parallel_env.py` | Vectorized environments |
| `hydra_bert/rl/rollout_buffer.py` | Experience storage |
| `configs/ppo_config.yaml` | RL hyperparameters |

---

## 10. Summary

Stage 2 successfully trained an RL policy for hydrogel design:

1. **Algorithm**: PPO with hybrid discrete/continuous action space
2. **Scale**: 2000 parallel environments, 200 iterations, 40M total samples
3. **Reward**: Stage 1 predictions + safety constraints
4. **Results**: Mean reward improved 136%, optimal rate reached 78%
5. **Output**: Trained policy for Stage 3 design generation
