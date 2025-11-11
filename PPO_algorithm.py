import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============================================================================
# ЧАСТЬ 1: ПРОСТАЯ СРЕДА CARTPOLE (упрощённая физика)
# ============================================================================

class SimpleCartPole:
    """
    Упрощённая версия CartPole для демонстрации PPO.
    
    Физика:
    - Тележка движется по горизонтальной оси
    - На ней закреплён шест, который нужно балансировать
    - Применяем силу влево (0) или вправо (1)
    """
    
    def __init__(self):
        self.gravity = 9.8          # g - ускорение свободного падения
        self.cart_mass = 1.0        # масса тележки
        self.pole_mass = 0.1        # масса шеста
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5      # половина длины шеста
        self.force_mag = 10.0       # сила толчка
        self.dt = 0.02              # шаг времени
        
        # Пороги для завершения эпизода
        self.x_threshold = 2.4      # тележка за пределами ±2.4
        self.theta_threshold = 12 * np.pi / 180  # угол > 12 градусов
        
        self.state = None
        self.steps = 0
        
    def reset(self) -> np.ndarray:
        """Сброс среды в случайное начальное состояние"""
        # Случайная инициализация близко к равновесию
        self.state = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Выполнить действие в среде
        
        Возвращает:
            state: новое состояние [x, x_dot, theta, theta_dot]
            reward: награда за шаг
            done: завершён ли эпизод
        """
        x, x_dot, theta, theta_dot = self.state
        
        # Сила: -10 (влево) или +10 (вправо)
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Физика: уравнения движения маятника на тележке
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Момент от гравитации и движения тележки
        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta) / self.total_mass
        
        # Угловое ускорение шеста: θ̈
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                    (self.pole_length * (4.0/3.0 - self.pole_mass * cos_theta**2 / self.total_mass))
        
        # Ускорение тележки: ẍ
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_theta / self.total_mass
        
        # Обновление состояния (интегрирование методом Эйлера)
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Проверка условий завершения
        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold or
            self.steps >= 500
        )
        
        # Награда: +1 за каждый шаг, пока система стабильна
        reward = 1.0
        
        return self.state.copy(), reward, done


# ============================================================================
# ЧАСТЬ 2: НЕЙРОННАЯ СЕТЬ НА ЧИСТОМ NUMPY
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU активация: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU"""
    return (x > 0).astype(float)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax для стабильных вероятностей
    
    Формула: σ(x_i) = exp(x_i) / Σ exp(x_j)
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class ActorCriticNetwork:
    """
    Комбинированная сеть Actor-Critic
    
    Actor (актёр): выдаёт вероятности действий π(a|s)
    Critic (критик): оценивает ценность состояния V(s)
    
    Архитектура:
        Input (4) → Hidden (64) → Hidden (64) → {Actor (2), Critic (1)}
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, lr: float = 3e-4):
        # Инициализация весов Xavier/He (с меньшей scale для стабильности)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim) * 0.5
        self.b2 = np.zeros(hidden_dim)
        
        # Actor head (голова актёра)
        self.W_actor = np.random.randn(hidden_dim, action_dim) * 0.01  # Маленькие веса для логитов
        self.b_actor = np.zeros(action_dim)
        
        # Critic head (голова критика)
        self.W_critic = np.random.randn(hidden_dim, 1) * 0.01  # Маленькие веса для value
        self.b_critic = np.zeros(1)
        
        self.lr = lr
        
        # Для хранения промежуточных значений (backprop)
        self.cache = {}
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Прямой проход
        
        Возвращает:
            action_probs: вероятности действий [π(a=0|s), π(a=1|s)]
            value: оценка V(s)
        """
        # Слой 1
        z1 = state @ self.W1 + self.b1
        a1 = relu(z1)
        
        # Слой 2
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        
        # Actor: логиты → softmax → вероятности
        logits = a2 @ self.W_actor + self.b_actor
        action_probs = softmax(logits)
        
        # Critic: линейный выход
        value = (a2 @ self.W_critic + self.b_critic)[0]
        
        # Сохраняем для backprop
        self.cache = {
            'state': state, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2, 'logits': logits,
            'action_probs': action_probs, 'value': value
        }
        
        return action_probs, value
    
    def backward(self, grad_logits: np.ndarray, grad_value: float):
        """
        Обратный проход (backpropagation) с gradient clipping
        
        grad_logits: градиент по логитам актёра
        grad_value: градиент по выходу критика
        """
        state = self.cache['state']
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        z1 = self.cache['z1']
        z2 = self.cache['z2']
        
        # === ГРАДИЕНТЫ ДЛЯ ACTOR HEAD ===
        grad_W_actor = a2.reshape(-1, 1) @ grad_logits.reshape(1, -1)
        grad_b_actor = grad_logits
        grad_a2_actor = grad_logits @ self.W_actor.T
        
        # === ГРАДИЕНТЫ ДЛЯ CRITIC HEAD ===
        grad_W_critic = a2.reshape(-1, 1) * grad_value
        grad_b_critic = np.array([grad_value])
        grad_a2_critic = self.W_critic.flatten() * grad_value
        
        # Суммируем градиенты от обеих голов
        grad_a2 = grad_a2_actor + grad_a2_critic
        
        # === СКРЫТЫЙ СЛОЙ 2 ===
        grad_z2 = grad_a2 * relu_derivative(z2)
        grad_W2 = a1.reshape(-1, 1) @ grad_z2.reshape(1, -1)
        grad_b2 = grad_z2
        grad_a1 = grad_z2 @ self.W2.T
        
        # === СКРЫТЫЙ СЛОЙ 1 ===
        grad_z1 = grad_a1 * relu_derivative(z1)
        grad_W1 = state.reshape(-1, 1) @ grad_z1.reshape(1, -1)
        grad_b1 = grad_z1
        
        # === GRADIENT CLIPPING (критично для стабильности!) ===
        max_grad_norm = 0.5
        
        grad_W1 = np.clip(grad_W1, -max_grad_norm, max_grad_norm)
        grad_b1 = np.clip(grad_b1, -max_grad_norm, max_grad_norm)
        grad_W2 = np.clip(grad_W2, -max_grad_norm, max_grad_norm)
        grad_b2 = np.clip(grad_b2, -max_grad_norm, max_grad_norm)
        grad_W_actor = np.clip(grad_W_actor, -max_grad_norm, max_grad_norm)
        grad_b_actor = np.clip(grad_b_actor, -max_grad_norm, max_grad_norm)
        grad_W_critic = np.clip(grad_W_critic, -max_grad_norm, max_grad_norm)
        grad_b_critic = np.clip(grad_b_critic, -max_grad_norm, max_grad_norm)
        
        # === ОБНОВЛЕНИЕ ВЕСОВ (SGD) ===
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W_actor -= self.lr * grad_W_actor
        self.b_actor -= self.lr * grad_b_actor
        self.W_critic -= self.lr * grad_W_critic
        self.b_critic -= self.lr * grad_b_critic


# ============================================================================
# ЧАСТЬ 3: PPO АЛГОРИТМ
# ============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Гиперпараметры:
        epsilon: порог клиппирования (обычно 0.2)
        gamma: коэффициент дисконтирования будущих наград
        lambda_gae: параметр для GAE (Generalized Advantage Estimation)
        c1: коэффициент value loss
        c2: коэффициент энтропии
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 epsilon: float = 0.2, gamma: float = 0.99, 
                 lambda_gae: float = 0.95, c1: float = 0.5, c2: float = 0.01):
        self.network = ActorCriticNetwork(state_dim, 64, action_dim, lr=1e-4)  # Уменьшили LR!
        
        self.epsilon = epsilon      # ε для клиппирования
        self.gamma = gamma          # γ дисконтирование
        self.lambda_gae = lambda_gae  # λ для GAE
        self.c1 = c1                # коэффициент value loss
        self.c2 = c2                # коэффициент энтропии
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Выбрать действие согласно текущей политике
        
        Возвращает:
            action: выбранное действие
            log_prob: логарифм вероятности действия
            value: оценка V(s)
        """
        action_probs, value = self.network.forward(state)
        
        # Защита от NaN
        if np.isnan(action_probs).any() or np.isnan(value):
            print("⚠️  Warning: NaN detected! Using random policy.")
            action_probs = np.ones(len(action_probs)) / len(action_probs)
            value = 0.0
        
        # Нормализация для численной стабильности
        action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        # Сэмплируем действие из распределения
        action = np.random.choice(len(action_probs), p=action_probs)
        
        # log π(a|s) - для вычисления ratio
        log_prob = np.log(action_probs[action] + 1e-8)
        
        return action, log_prob, value
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE)
        
        Формула:
            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD error)
            A_t = Σ (γλ)^k · δ_{t+k}
        
        GAE балансирует между:
        - Низкой дисперсией (λ→0): A = δ (1-step TD)
        - Низким смещением (λ→1): A = Σ γ^k·r (Monte Carlo)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        # Идём с конца траектории
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            # TD error: δ = r + γ·V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE: A = δ + (γλ)·A_{t+1}
            gae = delta + self.gamma * self.lambda_gae * gae
            
            advantages.insert(0, gae)
            
            # Return: G = A + V (advantage + baseline)
            returns.insert(0, gae + values[t])
            
            next_value = values[t]
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Нормализация advantages (стабилизирует обучение)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Нормализация returns для стабильности critic (ВАЖНО!)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               old_log_probs: np.ndarray, advantages: np.ndarray, 
               returns: np.ndarray, epochs: int = 10):
        """
        Обновление политики с помощью PPO
        
        Выполняем несколько эпох градиентного спуска по собранным данным
        """
        n_samples = len(states)
        
        for epoch in range(epochs):
            # Shuffling для лучшей сходимости
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                state = states[idx]
                action = actions[idx]
                old_log_prob = old_log_probs[idx]
                advantage = advantages[idx]
                return_target = returns[idx]
                
                # === FORWARD PASS ===
                action_probs, value = self.network.forward(state)
                
                # Логарифм вероятности текущего действия
                new_log_prob = np.log(action_probs[action] + 1e-8)
                
                # === PPO CLIPPED OBJECTIVE ===
                # Ratio: r = π_new / π_old = exp(log π_new - log π_old)
                ratio = np.exp(new_log_prob - old_log_prob)
                
                # Два варианта objective:
                # 1) r·A
                surr1 = ratio * advantage
                
                # 2) clip(r, 1-ε, 1+ε)·A
                surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                
                # Берём минимум (пессимистичная оценка)
                actor_loss = -np.minimum(surr1, surr2)
                
                # === VALUE FUNCTION LOSS (с клиппингом для стабильности) ===
                value_error = return_target - value
                value_error = np.clip(value_error, -10, 10)  # Ограничиваем ошибку!
                value_loss = self.c1 * value_error ** 2
                
                # === ENTROPY BONUS (для exploration) ===
                # H(π) = -Σ π(a|s)·log π(a|s)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                entropy_loss = -self.c2 * entropy
                
                # === TOTAL LOSS ===
                total_loss = actor_loss + value_loss + entropy_loss
                
                # === BACKWARD PASS ===
                # Градиент по логитам актёра
                grad_logits = action_probs.copy()
                
                # Для выбранного действия:
                if advantage > 0:
                    # Хорошее действие: увеличиваем вероятность (но с ограничением)
                    if ratio > 1 + self.epsilon:
                        grad_logits[action] = 0  # Не обновляем, если уже достаточно увеличили
                    else:
                        grad_logits[action] -= 1 / (action_probs[action] + 1e-8) * advantage
                else:
                    # Плохое действие: уменьшаем вероятность
                    if ratio < 1 - self.epsilon:
                        grad_logits[action] = 0
                    else:
                        grad_logits[action] -= 1 / (action_probs[action] + 1e-8) * advantage
                
                # Добавляем градиент энтропии (для exploration)
                grad_logits += self.c2 * (np.log(action_probs + 1e-8) + 1)
                
                # Градиент критика (с ограничением)
                grad_value = -2 * self.c1 * value_error
                grad_value = np.clip(grad_value, -1.0, 1.0)  # Клиппим градиент!
                
                # Обновляем веса
                self.network.backward(grad_logits, grad_value)


# ============================================================================
# ЧАСТЬ 4: ОБУЧЕНИЕ
# ============================================================================

def train_ppo(episodes: int = 500, max_steps: int = 500):
    """Основной цикл обучения PPO на CartPole"""
    env = SimpleCartPole()
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    episode_rewards = []
    
    print("🚀 Начинаем обучение PPO на CartPole!")
    print("=" * 60)
    
    for episode in range(episodes):
        state = env.reset()
        
        # Буферы для хранения траектории
        states, actions, rewards = [], [], []
        old_log_probs, values, dones = [], [], []
        
        episode_reward = 0
        
        # Собираем траекторию
        for step in range(max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Сохраняем в буфер
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Вычисляем advantages и returns
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        # Обновляем политику (меньше эпох для стабильности)
        agent.update(
            np.array(states),
            np.array(actions),
            np.array(old_log_probs),
            advantages,
            returns,
            epochs=5  # Уменьшили с 10 до 5
        )
        
        episode_rewards.append(episode_reward)
        
        # Логирование
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode + 1:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg(20): {avg_reward:6.1f}")
    
    return episode_rewards


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    rewards = train_ppo(episodes=30000)
    
    # График обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Raw rewards')
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), 
             label='Moving average (20)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (20 episodes)')
    plt.title('Smoothed Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ Обучение завершено!")
    print(f"Финальная средняя награда: {np.mean(rewards[-50:]):.1f}")
    print("=" * 60)
