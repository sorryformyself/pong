import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
from gym.envs.mspacman_array_state.Utils import Utils
from collections import deque

tf.get_logger().setLevel('ERROR')
episodes = 3000
episode_rewards = []
step_limit = 500000
memory_size = 100000
env = gym.make('PongDeterministic-v4')
env.seed(777)

print(env.observation_space)
state_size = [80, 80, 4]
action_size = 2

print(action_size)

# Episode 435	Average Score: 18.17	epsilon:0.01	per_beta: 1.00	frame: 962509
# Running for 4 hours 34 minutes 53 seconds
# Running for 924853 steps
def preprocess_frame(frame):
    # (210, 160, 1)
    frame = tf.image.rgb_to_grayscale(frame)
    # [Up: Down, Left: right]
    frame = frame[None, :]
    frame = tf.keras.layers.Cropping2D(cropping=((34, 16), (0, 0)))(frame)
    frame = tf.squeeze(frame, 0)

    frame = tf.image.resize(frame, [80, 80])
    frame = tf.squeeze(frame, 2)
    frame = frame / 255

    frame = tf.image.convert_image_dtype(frame, dtype=tf.float16)
    return frame.numpy()  # 80x80 frame


stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque(maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames.clear()
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
    stacked_frames = np.stack(stacked_frames, axis=2)
    return stacked_frames


# leaves contain priorities for every experience.A data array containing the experiences points to the leaves.
# priorities are determined due to their TD error.
# Updating the tree and sampling will be really efficient (O(log n)).
# the value of root node is the sum of its child nodes
class SumTree:
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, priority, data):
        # the overall nodes is capacity(leaves) + capacity - 1
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        changed_value = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # if index = 6, then index = 2 and index = 0 in tree will add changed_value
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += changed_value

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    # an alternative method for getter and setter
    # we could directly use self.total_priority instead of self.get_total_priority()
    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class DQNAgent:
    def __init__(self):
        # other hyperparameters
        self.save_graph = True
        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.load_model = False
        self.random = False
        self.dueling = True
        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 500000
        self.decay_rate = 0.995

        # fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 1000
        self.target_network_counter = 0

        # n-step learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # experience replay used SumTree
        # combine agent and PER
        self.batch_size = 32
        self.gamma = 0.99
        self.replay_start_size = 32
        self.experience_replay = SumTree(memory_size)
        self.PER_e = 0.01  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
        self.PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a
        self.PER_b = 0.4
        self.PER_b_increment = 0.002
        self.absolute_error_upper = 1.  # clipped error
        self.experience_number = 0
        # initially, p1=1 total_priority=1,so P(1)=1,w1=batchsize**beta

        # check the hyperparameters
        if self.random:
            self.play = False
            self.isTraining = False
        if self.play:
            self.render = True
            self.load_model = True
            self.isTraining = False
            self.keepTraining = False
            self.epsilon = 0
        if self.keepTraining:
            self.epsilon = self.min_epsilon
            self.PER_b = 1
            self.load_model = True

        if self.load_model:
            self.model = keras.models.load_model('pong_model.h5')
            self.target_model = keras.models.load_model('pong_model.h5')
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

    # n-step learning, get the truncated n-step return
    def get_n_step_info(self, n_step_buffer, gamma):
        """Return n step reward, next state, and done."""
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    # these three methods:sample,store,batch_update are used in experience replay
    def sample(self, n):
        mini_batch = []
        batch_index = np.empty((n,), dtype=int)
        batch_ISWeights = np.empty((n,), dtype=float)
        priority_segment = self.experience_replay.total_priority / n
        # if self.PER_b < 1:
        #     self.PER_b += self.PER_b_increment

        min_priority_probability = np.min(
            self.experience_replay.tree[-self.experience_replay.capacity:]) / self.experience_replay.total_priority
        if min_priority_probability == 0:
            min_priority_probability = 1 / memory_size
        # max_weight = (min_priority_probability * memory_size) ** (-self.PER_b)
        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.experience_replay.get_leaf(value)
            sampling_probability = priority / self.experience_replay.total_priority
            # batch_ISWeights[i] = np.power(sampling_probability*memory_size,-self.PER_b) / max_weight
            batch_ISWeights[i] = np.power(sampling_probability / min_priority_probability, -self.PER_b)
            batch_index[i] = index
            mini_batch.append(data)
        return batch_index, mini_batch, batch_ISWeights

    # newly transitions have max_priority or 1 at first transition
    def store(self, experience):
        max_priority = np.max(self.experience_replay.tree[-self.experience_replay.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        if self.experience_number < memory_size:
            self.experience_number += 1

        # n_step
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
            state, action = self.n_step_buffer[0][:2]
            self.experience_replay.add(max_priority, (state, action, reward, next_state, done))

    def batch_update(self, tree_index, abs_errors):
        abs_errors = tf.add(abs_errors, self.PER_e)
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.PER_a)
        for index, priority in zip(tree_index, priorities):
            self.experience_replay.update(index, priority)

    # DDDQN dueling double DQN, the network structure should change
    def create_model(self):
        inputs = tf.keras.Input(shape=state_size)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8,
                                       strides=4, padding="VALID", activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
                                       strides=2, padding="VALID", activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                       strides=1, padding="VALID", activation='relu')(conv2)
        flatten = tf.keras.layers.Flatten()(conv3)
        fc1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
        # fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc1)
        if self.dueling:
            value_out = tf.keras.layers.Dense(1, activation='linear')(fc1)
            norm_advantage_output = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
            # outputs = tf.keras.layers.Add()([value_out,advantage_output-tf.reduce_mean(advantage_output,axis=1,keepdims=True)])
            outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
            model = tf.keras.Model(inputs, outputs)
        else:
            model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.summary()
        return model
    # 10700 13.8g

    @Utils.lp_wrapper()
    def training(self):
        if self.experience_number >= self.replay_start_size:
            # if self.epsilon > self.min_epsilon:
            #    self.epsilon = self.epsilon * self.decay_rate
            # batches = random.sample(self.experience_replay, self.batch_size)
            batch_index, batches, batch_ISWeights = self.sample(self.batch_size)
            absolute_errors = []
            buffer_state = np.vstack([data[0] for data in batches])
            buffer_action = [data[1] for data in batches]
            buffer_reward = [data[2] for data in batches]
            buffer_next_state = np.vstack([data[3] for data in batches])
            buffer_done = [data[4] for data in batches]

            n_gamma = self.gamma ** self.n_step
            y = self.local_inference(buffer_state).numpy()
            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.local_inference(buffer_next_state).numpy(), axis=1)
            target_y = self.target_inference(buffer_next_state).numpy()

            target_network_q_value = target_y[np.arange(self.batch_size), max_action_next]

            q_values_req = np.where(buffer_done, buffer_reward, buffer_reward + n_gamma * target_network_q_value)
            absolute_errors = tf.abs(y[np.arange(self.batch_size), buffer_action] - q_values_req)
            y[np.arange(self.batch_size), buffer_action] = q_values_req
            # n_step learning: gamma is also truncated
            # now the experience actually store n-step info
            # such as state[0], action[0], n-step reward, next_state[2] and done[2]

            #history = self.model.fit(buffer_state, y, batch_size=self.batch_size, epochs=1, verbose=0,
            #                         sample_weight=batch_ISWeights)
            # history = self.model.train_on_batch(buffer_state, y, sample_weight=batch_ISWeights)
            history = self.train_batch(buffer_state, y, batch_ISWeights)
            # 270053.0     51.7 tf.function, train_batch
            # 657247.0     70.8 without tf.function, train_batch
            # 372649.0     58.3 train_on_batch

            # gpu: 37549 17.6 tf.function, train_batch
            # 180044 56.9% without tf.function
            # 192116 57.4% train_on_batch
            # 679157 83% model.fit
            self.batch_update(batch_index, absolute_errors)
            return history

    @tf.function
    def local_inference(self, x):
        return self.model(x, training=False)

    @tf.function
    def target_inference(self, x):
        return self.target_model(x, training=False)

    @tf.function
    def train_batch(self, x, y, sample_weight):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss(y, predictions, sample_weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            # print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.local_inference(state).numpy()[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        return action + 2

    def draw(self, rewards, location):
        plt.plot(rewards)
        plt.title('score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Last Score')
        plt.ylim(bottom=-22)
        plt.savefig(location)
        plt.close()


agent = DQNAgent()

if agent.isTraining:
    scores_window = deque(maxlen=30)
    start = time.time()
    for episode in range(1, episodes + 1):
        rewards = 0
        state = env.reset()
        state = stack_frames(stacked_frames, state, True)
        state = state[None, :]
        while True:
            # 2 up 3 down
            action = agent.acting(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = stack_frames(stacked_frames, next_state, False)
            next_state = next_state[None, :]
            reward = -10 if done else reward

            agent.store((state, action - 2, reward, next_state, done))
            state = next_state
            history = agent.training()
            if done or rewards >= step_limit:
                next_state = stack_frames(stacked_frames, np.zeros((210, 160, 3), dtype=np.int), False)
                next_state = next_state[None, :]
                agent.store((state, action - 2, reward, next_state, done))
                episode_rewards.append(rewards)
                scores_window.append(rewards)
                if agent.PER_b < 1:
                    agent.PER_b += agent.PER_b_increment
                break
        print('\rEpisode {}\tAverage Score: {:.2f}\tepsilon:{:.2f}\tper_beta: {:.2f}\tframe: {}'.format(episode,
                                                                                             np.mean(scores_window),
                                                                                             agent.epsilon,
                                                                                             agent.PER_b, agent.target_network_counter), end="")

        if np.mean(scores_window) > 20:
            Utils.printSolvedTime(start, episode)
            agent.model.save('pong_model.h5')
            Utils.saveRewards('pong', episode_rewards, agent.keepTraining, 30)
            Utils.saveThreePlots('pong', episode_rewards, agent.keepTraining, bottom=-22)
            break
        if episode % 30 == 0:
            Utils.printRunningTime(start, agent.target_network_counter)
            agent.model.save('pong_model.h5')
            Utils.saveRewards('pong', episode_rewards, agent.keepTraining, 30)
            Utils.saveThreePlots('pong', episode_rewards, agent.keepTraining, bottom=-22)
    env.close()

if agent.play:
    scores_window = deque(maxlen=30)
    for episode in range(1, 11):
        rewards = 0
        state = env.reset()

        state = stack_frames(stacked_frames, state, True)
        state = state[None, :]
        while True:
            # 2 up 3 down
            action = agent.acting(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = stack_frames(stacked_frames, next_state, False)
            next_state = next_state[None, :]
            state = next_state
            if done:
                scores_window.append(rewards)
                break
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
    env.close()
