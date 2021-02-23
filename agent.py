from collections import deque

import numpy as np
import tensorflow as tf
from gym.envs.mspacman_array_state.Utils import Utils

state_size = (84, 84, 4)
action_size = 4


def set_weights_fn(policy, weights):
    policy.model.set_weights(weights)


saveFileName = 'gym_pong'


class Agent:
    def __init__(self):
        # other hyper parameters

        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.load_model = False
        self.random = False
        self.dueling = True

        # fixed q value - two networks
        self.learning_rate = 0.00025 / 4  # parameter in apex paper
        self.fixed_q_value_steps = 2500  # parameter in apex paper
        self.target_network_counter = 0

        # experience replay used SumTree
        # combine agent and PER
        self.batch_size = 512
        self.gamma = 0.99
        self.n_warmup = 3200
        self.PER_e = 1e-6  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
        self.PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a
        self.PER_b = 0.4

        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon

        # n-step learning
        self.n_step = 5
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.n_gamma = self.gamma ** self.n_step

        if self.load_model:
            self.model = tf.keras.models.load_model(saveFileName + '_completed.h5')
            self.target_model = tf.keras.models.load_model(saveFileName + '_completed.h5')
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

    def create_model(self):
        inputs = tf.keras.Input(shape=state_size)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8,
                                       strides=4, padding="valid", activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
                                       strides=2, padding="valid", activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                       strides=1, padding="valid", activation='relu')(conv2)
        flatten = tf.keras.layers.Flatten()(conv3)
        fc1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
        fc2 = tf.keras.layers.Dense(512, activation='relu')(flatten)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc1)
        if self.dueling:
            value_out = tf.keras.layers.Dense(1, activation='linear')(fc2)
            norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
            outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
            model = tf.keras.Model(inputs, outputs)
        else:
            model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done, weights):
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        td_errors, loss = self._train_body(buffer_state, buffer_action,
                                           buffer_reward, buffer_next_state, buffer_done, weights)
        # tf.summary.scalar(name='learner ' + "/model_Loss", data=loss)
        return td_errors

    def compute_td_error(self, buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done):
        return self._compute_td_error_body(buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done)

    def explorer_compute_td_error(self, buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done):
        return self._explorer_compute_td_error_body(buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done)

    @tf.function
    def _train_body(self, states, actions, rewards, next_states, dones, weights):
        # weights = tf.cast(weights, dtype=tf.float16)
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(states, actions, rewards, next_states, dones)
            # loss = tf.reduce_mean(self.huber_loss(td_errors, delta=10.) * weights)
            loss = tf.reduce_mean(tf.square(td_errors) * weights)  # huber loss seems no use
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return td_errors, loss

    @tf.function
    def _compute_td_error_body(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        states = states / 255
        next_states = next_states / 255
        rewards = tf.cast(tf.squeeze(rewards), dtype=tf.float32)
        dones = tf.cast(tf.squeeze(dones), dtype=tf.bool)
        actions = tf.cast(actions, dtype=tf.int32)  # (batch_size, 1)
        batch_size_range = tf.expand_dims(tf.range(batch_size), axis=1)  # (batch_size, 1)

        # get current q value
        current_q_indexes = tf.concat(values=(batch_size_range, actions), axis=1)  # (batch_size, 2)
        current_q = tf.gather_nd(self.model(states), current_q_indexes)  # (batch_size, )

        # get target q value using double dqn
        max_next_q_indexes = tf.argmax(self.model(next_states), axis=1, output_type=tf.int32)  # (batch_size, )
        indexes = tf.concat(values=(batch_size_range,
                                    tf.expand_dims(max_next_q_indexes, axis=1)), axis=1)  # (batch_size, 2)
        target_q = tf.gather_nd(self.target_model(next_states), indexes)  # (batch_size, )
        target_q = tf.where(dones, rewards, rewards + self.n_gamma * target_q)  # (batch_size, )

        # don't want change the weights of target network in backpropagation, so tf.stop_gradient()
        # but seems no use
        td_errors = tf.abs(current_q - tf.stop_gradient(target_q))
        return td_errors

    @tf.function
    def _explorer_compute_td_error_body(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        states = states / 255
        next_states = next_states / 255
        rewards = tf.cast(tf.squeeze(rewards), dtype=tf.float32)
        dones = tf.cast(tf.squeeze(dones), dtype=tf.bool)
        actions = tf.cast(actions, dtype=tf.int32)  # (batch_size, 1)
        batch_size_range = tf.expand_dims(tf.range(batch_size), axis=1)  # (batch_size, 1)

        # get current q value
        current_q_indexes = tf.concat(values=(batch_size_range, actions), axis=1)  # (batch_size, 2)
        current_q = tf.gather_nd(self.model(states), current_q_indexes)  # (batch_size, )

        # get target q value using double dqn
        max_next_q_indexes = tf.argmax(self.model(next_states), axis=1, output_type=tf.int32)  # (batch_size, )
        indexes = tf.concat(values=(batch_size_range,
                                    tf.expand_dims(max_next_q_indexes, axis=1)), axis=1)  # (batch_size, 2)
        target_q = tf.gather_nd(self.model(next_states), indexes)  # (batch_size, )
        target_q = tf.where(dones, rewards, rewards + self.n_gamma * target_q)  # (batch_size, )

        # don't want change the weights of target network in backpropagation, so tf.stop_gradient()
        # but seems no use
        td_errors = tf.abs(current_q - tf.stop_gradient(target_q))
        return td_errors

    def huber_loss(self, x, delta=1.):
        """

        Args:
            x: np.ndarray or tf.Tensor
                Values to compute the huber loss.
            delta: float
                Positive floating point value. Represents the
                maximum possible gradient magnitude.

        Returns: tf.Tensor
            The huber loss.
        """
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x)
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
        return tf.where(
            tf.abs(x) <= delta,
            x=less_than_max,
            y=greater_than_max)

    def acting(self, state, test=False):
        if not test:
            if np.random.sample() <= self.epsilon:
                return np.random.randint(action_size)
        return self._get_action_body(state).numpy()

    @tf.function
    def _get_action_body(self, state):
        state = state / 255
        state = tf.expand_dims(state, axis=0)
        qvalues = self.model(state)[0]
        return tf.argmax(qvalues)


if __name__ == '__main__':
    import gym

    agent = Agent()
    agent.model = tf.keras.models.load_model('20210112T141401.891389_evaluator_model_15minMax',
                                             custom_objects={'tf': tf})
    agent.target_model = tf.keras.models.load_model('20210112T141401.891389_evaluator_model_15minMax',
                                                    custom_objects={'tf': tf})
    scores_window = deque(maxlen=30)
    episode_rewards = []
    env = gym.make('PongDeterministic-v4')
    stacked_frames = deque(maxlen=4)
    from apex import stack_frames
    import time
    for episode in range(1, 6):
        rewards = 0
        state = env.reset()
        state = stack_frames(stacked_frames, state, True)
        while True:
            action = agent.acting(state, test=True)
            env.render()
            time.sleep(0.02)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = stack_frames(stacked_frames, next_state, False)
            state = next_state
            if done:
                scores_window.append(rewards)
                break
        Utils.print(episode, scores_window, agent.target_network_counter)

    env.close()
