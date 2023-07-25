import os
import gymnasium as gym
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# # # PROBLEM DEFINITION # # #
problem = "CarRacing-v2"
env = gym.make(problem, render_mode='human')
env.reset()

num_states = 96*96
print("Size of State Space ->  {}".format(num_states))

num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# Wout = (Win - K + 2*P)/S + 1
# Hout = (Hin - K + 2*P)/S + 1.
# input: Win x Hin
# kernel: KxK
# stride: S
# padding: P


class GaussianNoise:
    def __init__(self, mean, std_deviation):
        self.mean = mean
        self.std_dev = std_deviation

    def __call__(self):
        x = np.random.normal(self.mean, self.std_dev)
        return x


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on
        self.batch_size = batch_size
        # It tells us the number of times record() was called
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, 96, 96))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions-1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 96, 96))

    # Takes (s, a, r, s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed-up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, d_sw):
        # Training and updating Actor & Critic network
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * (1 - d_sw) * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

# We compute the loss and update parameters
    def learn(self, d_sw):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, d_sw)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # Batch Normalized
    inputs = layers.Input(shape=(96, 96, 1))
    # convolution start
    out = layers.Conv2D(filters=4, kernel_size=4, strides=(4, 4))(inputs)
    out = layers.Conv2D(filters=8, kernel_size=4, strides=(2, 2))(out)
    out = layers.Conv2D(filters=12, kernel_size=4, strides=(2, 2))(out)
    out = layers.Flatten()(out)
    # convolution end
    out = layers.BatchNormalization()(out)
    out = layers.Dense(400)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    out = layers.Dense(500)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    # 3 is the number of actions: beware! We usually never want to brake and accelerate at the same time!
    # 3 number of actions: put 2. In this way, the tanh function is perfect
    # First output is steering, while second is: out2 > 0: throttle, out2 < 0: brake!
    out = layers.Dense(2, kernel_initializer=last_init)(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Activation('tanh')(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # Batch Normalized
    # State as input
    state_input = layers.Input(shape=(96, 96, 1))  # shape=num_states if no convolution
    # convolution start
    out = layers.Conv2D(filters=4, kernel_size=4, strides=(4, 4))(state_input)  # (23x23) x 8
    out = layers.Conv2D(filters=8, kernel_size=4, strides=(2, 2))(out)  # (11x11) x 4
    out = layers.Conv2D(filters=12, kernel_size=4, strides=(2, 2))(out)
    out = layers.Flatten()(out)
    # convolution end
    state_out = layers.BatchNormalization()(out)
    state_out = layers.Dense(400)(state_out)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Activation('relu')(state_out)
    state_out = layers.Dense(500)(state_out)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Activation('relu')(state_out)

    # Action as input
    action_input = layers.Input(shape=num_actions-1)
    action_out = layers.BatchNormalization()(action_input)
    action_out = layers.Dense(64)(action_out)
    action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Activation('relu')(action_out)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object, steps_number):

    if steps_number < start_steps:
        sampled_actions = np.random.uniform(low=[-1, -1], high=[1, 1])
    else:
        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    # Must make corrections, because we have more than one actions here
    # In this way, we prevent the car from braking and accelerating simultaneously
    # legal_action = np.zeros(3)
    legal_action = 0*np.ndarray((3,))
    legal_action[0] = sampled_actions[0]
    if sampled_actions[1] >= 0:
        legal_action[1] = sampled_actions[1]
    else:
        legal_action[2] = abs(sampled_actions[1])

    legal_action = np.clip(legal_action, lower_bound, upper_bound)
    # no sampled_actions, we must clip it, too
    sa_upper_bound = np.ones((2,))
    sa_lower_bound = -np.ones((2,))
    legal_sampled_actions = np.clip(sampled_actions, sa_lower_bound, sa_upper_bound)

    return [np.squeeze(legal_action), legal_sampled_actions]


class ThrottleBuffer:
    def __init__(self, throttle_buffer_capacity=400, epsilon=1e-1):
        self.bufferCapacity = throttle_buffer_capacity
        self.epsilon = epsilon
        self.buffer = np.ones((self.bufferCapacity,))
        self.bufferIdx = 0

    def record(self, throttle):
        tmp_buffer_idx = self.bufferIdx % self.bufferCapacity
        self.buffer[tmp_buffer_idx] = throttle
        self.bufferIdx += 1

    def is_not_moving(self):
        not_moving = np.mean(abs(self.buffer)) < self.epsilon  # sum(self.buffer) < self.epsilon
        return not_moving

    def reset(self):
        self.buffer = np.ones((self.bufferCapacity,))
        self.bufferIdx = 0


std_dev = 0.2
# action_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
action_noise = GaussianNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

files = os.listdir('./')

for name in files:
    if name == 'racingCar_actor.h5':
        print('Loading actor weights...')
        actor_model.load_weights(name)
        print('... actor weights loaded')
    elif name == 'racingCar_critic.h5':
        print('Loading critic weights...')
        critic_model.load_weights(name)
        print('... critic weights loaded')
    elif name == 'racingCar_target_actor.h5':
        print('Loading target actor weights...')
        target_actor.load_weights(name)
        print('... target actor weights loaded')
    elif name == 'racingCar_target_critic.h5':
        print('Loading target critic weights...')
        target_critic.load_weights(name)
        print('... target critic weights loaded')

# Learning rate for actor-critic models
critic_lr = 0.00095
actor_lr = 0.00085

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

maxInitialEpisodeTime = 5
maxEpisodeTime = maxInitialEpisodeTime

n_steps = 0
start_steps = 1
update_every = 5

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

throttleBufferLength = 400
eps = 1e-1
throttleBuffer = ThrottleBuffer(throttleBufferLength, eps)

best_reward = -np.inf
os.mkdir('./best_model')

print('Starting to train')

for ep in range(total_episodes):

    prev_state = env.reset()
    prev_state = prev_state[0]
    prev_state = 0.299*prev_state[:, :, 0] + 0.587*prev_state[:, :, 1] + 0.114*prev_state[:, :, 2]  # 0.299*red + 0.587*green + 0.114*blue
    episodic_reward = 0

    startingTime = time.time()  # DEBUG ADDITION

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action, actionForBuffer = policy(tf_prev_state, action_noise, n_steps)
        throttleBuffer.record(action[1])
        result = env.step(action)
        state = result[0]
        state = 0.299*state[:, :, 0] + 0.587*state[:, :, 1] + 0.114*state[:, :, 2]
        reward = result[1]
        done = result[2]
        info = result[3]

        buffer.record((prev_state, actionForBuffer, reward, state))  # buffer records reshaped state
        episodic_reward += reward

        currentTime = time.time()
        diffTime = currentTime - startingTime
        overcomeTime = diffTime > maxEpisodeTime
        notMoving = throttleBuffer.is_not_moving()
        d = done or overcomeTime or notMoving

        if n_steps % update_every == 0:
            buffer.learn(int(d))
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

        env.render()
        n_steps += 1

        # End this episode when 'd' is True
        if d:
            print('best_reward is', best_reward)
            print('episodic_reward is', episodic_reward)
            maxEpisodeTime = maxInitialEpisodeTime + 0.00005*(ep+1 - 1)**2
            throttleBuffer.reset()
            if episodic_reward > best_reward:
                print('Found the best model so far')
                actor_model.save_weights('./best_model/racingCar_actor.h5')
                critic_model.save_weights('./best_model/racingCar_critic.h5')
                target_actor.save_weights('./best_model/racingCar_target_actor.h5')
                target_critic.save_weights('./best_model/racingCar_target_critic.h5')
                best_reward = episodic_reward
            print('Done')
            break

        prev_state = state

    print('Episode number ', ep + 1)
    print('Episode reward ', episodic_reward)
    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    actor_model.save_weights("racingCar_actor.h5")
    critic_model.save_weights("racingCar_critic.h5")
    target_actor.save_weights("racingCar_target_actor.h5")
    target_critic.save_weights("racingCar_target_critic.h5")

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
