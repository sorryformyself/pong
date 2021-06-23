import datetime
import multiprocessing
import os
import threading
import time
from collections import deque
from multiprocessing import Process, Event, Value
from multiprocessing.managers import SyncManager

import cv2
import gym
import numpy as np
from agent import Agent, get_weights_fn, set_weights_fn
from cpprb import PrioritizedReplayBuffer, ReplayBuffer
from gym.envs.mspacman_array_state.Utils import Utils

step_limit = 50000
memory_size = 200000

state_size = (84, 84, 4)
action_size = 4

frame_skip = 4  # return one frame in every four frame
# Initialize deque with zero-images one array for each image

saveFileName = 'gym_pong'
saveInternal = 50


def import_tf():
    import tensorflow as tf
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    return tf


class env_fn:
    def __init__(self, name):
        self.env_name = name

    def __call__(self):
        return gym.make(self.env_name)


_env = env_fn('PongDeterministic-v4')


def preprocess_frame(frame):
    image = frame

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (state_size[0], state_size[1]), interpolation=cv2.INTER_AREA)
    image = image / 255
    image = image.astype(np.float16, copy=False)

    return image


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


# not used now. Edited in gym
def frame_skip_step(env, action, video=None):
    total_reward = 0.0
    for i in range(frame_skip):
        if video:
            Utils.writeOneFrame(video, env.render())
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return obs, total_reward, done, info


def explorer(global_rb, queue, trained_steps, is_training_done,
             lock, buffer_size=1024, episode_max_steps=1000, epsilon=0.5, transitions=None):
    tf = import_tf()
    env = _env()
    stacked_frames = deque(maxlen=4)
    policy = Agent()
    policy.epsilon = epsilon
    env_dict = {"obs": {"shape": state_size},
                "act": {},
                "rew": {},
                "next_obs": {"shape": state_size},
                "done": {}}
    local_rb = ReplayBuffer(buffer_size, env_dict=env_dict, default_dtype=np.float16)
    local_idx = np.arange(buffer_size).astype(np.int)

    s = env.reset()
    s = stack_frames(stacked_frames, s, True)
    episode_steps = 0
    total_reward = 0.
    total_rewards = []

    start = time.time()
    n_sample, n_sample_old = 0, 0

    while not is_training_done.is_set():
        transitions.value += 1
        n_sample += 1
        episode_steps += 1
        a = policy.acting(s)
        s_, r, done, _ = env.step(a)
        done_flag = done
        if episode_steps == episode_max_steps:
            done_flag = False
        total_reward += r
        s_ = stack_frames(stacked_frames, s_, False)
        policy.n_step_buffer.append((s, a, r, s_, done_flag))
        if len(policy.n_step_buffer) == policy.n_step:
            reward, next_state, done = policy.get_n_step_info(policy.n_step_buffer, policy.gamma)
            state, action = policy.n_step_buffer[0][:2]
            local_rb.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

        s = s_
        if done or episode_steps == episode_max_steps:
            s = env.reset()
            s = stack_frames(stacked_frames, s, True)
            total_rewards.append(total_reward)
            total_reward = 0
            episode_steps = 0

        if not queue.empty():
            set_weights_fn(policy, queue.get())

        if local_rb.get_stored_size() == buffer_size:
            samples = local_rb._encode_sample(local_idx)

            samples1 = {key: value[:50] for key, value in samples.items()}
            samples2 = {key: value[50:100] for key, value in samples.items()}
            samples3 = {key: value[100:150] for key, value in samples.items()}
            samples4 = {key: value[150:200] for key, value in samples.items()}

            for samples in [samples1, samples2, samples3, samples4]:
                td_errors = policy.compute_td_error(
                    samples["obs"], samples["act"], samples["rew"],
                    samples["next_obs"], samples["done"])
                priorities = td_errors.numpy() + 1e-6
                samples['priority'] = priorities

            samples = {key: np.concatenate((value, samples2[key], samples3[key], samples4[key])) for key, value in
                       samples1.items()}

            global_rb.add(
                obs=samples["obs"], act=samples["act"], rew=samples["rew"],
                next_obs=samples["next_obs"], done=samples["done"],
                priorities=samples['priority'])
            local_rb.clear()

            ave_rew = (0 if len(total_rewards) == 0 else
                       sum(total_rewards) / len(total_rewards))

            total_rewards = []
            start = time.time()
            n_sample_old = n_sample


def sample(lock, global_rb, batch_size, tf_queue):
    while True:
        lock.acquire()
        samples = global_rb.sample(batch_size)
        lock.release()
        tf_queue.enqueue(samples)


def update_priority(lock, global_rb, update_priority_queue):
    while True:
        samples = update_priority_queue.dequeue()
        indexes, td_errors = samples['indexes'], samples['td_errors']
        lock.acquire()
        global_rb.update_priorities(indexes, td_errors)
        lock.release()


def learner(global_rb, trained_steps, is_training_done,
            lock, n_training, update_freq, evaluation_freq, queues, transitions, n_warmup=3200,
            batch_size=64):
    tf = import_tf()
    # Wait until explorers collect transitions
    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="learner")
    writer = tf.summary.create_file_writer(output_dir)
    policy = Agent()
    writer.set_as_default()
    if os.name == 'posix':
        os.system('cp apex.py ' + output_dir + '/apex.py')
        os.system('cp agent.py ' + output_dir + '/agent.py')

    while not is_training_done.is_set() and global_rb.get_stored_size() < n_warmup:
        continue

    start_time = time.time()
    n_transition = 0
    tf_queue = tf.queue.FIFOQueue(1, dtypes=[np.float16, np.float16, np.float16, np.float16, np.float16, np.float16,
                                             np.float32],
                                  names=['act', 'done', 'indexes', 'next_obs', 'obs', 'rew', 'weights'])
    t = threading.Thread(target=sample, args=(lock, global_rb, batch_size, tf_queue))
    t.start()

    update_priority_queue = tf.queue.FIFOQueue(1, dtypes=[np.float16, np.float16],
                                               names=['indexes', 'td_errors'])
    priority_thread = threading.Thread(target=update_priority, args=(lock, global_rb, update_priority_queue))
    priority_thread.start()

    while not is_training_done.is_set():

        # tf.summary.experimental.set_step(trained_steps.value)
        # lock.acquire()
        # samples = global_rb.sample(batch_size)
        # lock.release()
        # queues[-2].put(pickle.dumps(samples))

        trained_steps.value += 1
        samples = tf_queue.dequeue()
        td_errors = policy.train(
            samples["obs"], samples["act"], samples["rew"],
            samples["next_obs"], samples["done"], samples["weights"])

        # update_priority_queue.enqueue({'indexes': samples['indexes'], 'td_errors': td_errors.numpy()+1e-6})
        lock.acquire()
        global_rb.update_priorities(
            samples["indexes"], td_errors.numpy() + 1e-6)
        lock.release()

        # Put updated weights to queue
        if trained_steps.value % update_freq == 0:
            weights = get_weights_fn(policy)
            for i in range(len(queues) - 1):
                queues[i].put(weights)
            training_steps_per_second = update_freq / (time.time() - start_time)
            explorers_transitions_per_second = (transitions.value - n_transition) / (time.time() - start_time)
            explorers_frames_per_second = explorers_transitions_per_second * frame_skip
            steps = trained_steps.value

            tf.summary.scalar(name="apex/training_steps_per_second", data=training_steps_per_second, step=steps)
            tf.summary.scalar(name="apex/explorers_transitions_per_second",
                              data=explorers_transitions_per_second,
                              step=steps)
            tf.summary.scalar(name="apex/explorers_frames_per_second", data=explorers_frames_per_second,
                              step=steps)

            start_time = time.time()
            n_transition = transitions.value

        # Periodically do evaluation
        if trained_steps.value % evaluation_freq == 0:
            # writer.flush()
            queues[-1].put(get_weights_fn(policy))
            queues[-1].put(trained_steps.value)

        if trained_steps.value >= n_training:
            is_training_done.set()


def evaluator(is_training_done, queue,
              save_model_interval=int(1e6), n_evaluation=10, episode_max_steps=1000,
              show_test_progress=False):
    """

    @param is_training_done:
    @param queue:
    @type queue: multiprocessing.managers.Queue
    @param save_model_interval:
    @param n_evaluation:
    @param episode_max_steps:
    @param show_test_progress:
    """
    tf = import_tf()
    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="evaluator")
    writer = tf.summary.create_file_writer(
        output_dir, filename_suffix="_evaluation")
    writer.set_as_default()
    env = _env()
    stacked_frames = deque(maxlen=4)
    policy = Agent()
    model_save_threshold = save_model_interval

    while not is_training_done.is_set():
        n_evaluated_episode = 0
        # Wait until a new weights comes
        if queue.empty():
            continue
        else:
            set_weights_fn(policy, queue.get())
            trained_steps = queue.get()

            tf.summary.experimental.set_step(trained_steps)
            avg_test_return = 0.
            for _ in range(n_evaluation):
                n_evaluated_episode += 1
                episode_return = 0.
                obs = env.reset()
                obs = stack_frames(stacked_frames, obs, True)
                done = False
                for _ in range(episode_max_steps):
                    action = policy.acting(obs, test=True)
                    # next_obs, reward, done, _ = frame_skip_step(env, action)
                    next_obs, reward, done, _ = env.step(action)
                    if show_test_progress:
                        env.render()
                    episode_return += reward
                    next_obs = stack_frames(stacked_frames, next_obs, False)
                    obs = next_obs
                    if done:
                        break
                avg_test_return += episode_return
                # Break if a new weights comes
                if not queue.empty():
                    break
            avg_test_return /= n_evaluated_episode
            tf.summary.scalar(
                name="apex/average_test_return", data=avg_test_return)
            # writer.flush()
            if trained_steps > model_save_threshold:
                model_save_threshold += save_model_interval
                policy.model.save(output_dir + '_model')
                # checkpoint_manager.save()
    # checkpoint_manager.save()
    policy.model.save(output_dir + '_model')


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S.%f', suffix=""):
    if suffix is not "":
        suffix = "_" + suffix
    time_str = datetime.datetime.now().strftime(time_format) + suffix
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        raise RuntimeError('directory not specified')
    return outdir


if __name__ == '__main__':
    SyncManager.register('PrioritizedReplayBuffer',
                         PrioritizedReplayBuffer)
    manager = SyncManager()
    manager.start()

    PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a

    env_dict = {"obs": {"shape": state_size},
                "act": {},
                "rew": {},
                "next_obs": {"shape": state_size},
                "done": {}}
    global_rb = manager.PrioritizedReplayBuffer(memory_size, env_dict=env_dict, alpha=PER_a, default_dtype=np.float16,
                                                check_for_update=True)

    n_explorer = multiprocessing.cpu_count() - 1
    epsilons = [pow(0.4, 1 + (i / (n_explorer - 1)) * 7) for i in range(n_explorer)]  # apex paper

    n_queue = n_explorer
    n_queue += 1  # for evaluation
    # n_queue += 1  # for prefetch
    queues = [manager.Queue() for _ in range(n_queue)]

    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    is_training_done = Event()

    transitions = Value('i', 0)

    # Lock
    lock = manager.Lock()

    # Shared memory objects to count number of samples and applied gradients
    trained_steps = Value('i', 0)

    tasks = []
    local_buffer_size = 200  # 100论文数据
    episode_max_steps = step_limit

    for i in range(n_explorer):
        task = Process(
            target=explorer,
            args=[global_rb, queues[i], trained_steps, is_training_done,
                  lock, local_buffer_size, episode_max_steps, epsilons[i], transitions])
        task.start()
        tasks.append(task)

    n_training = 1000000
    param_update_freq = 200  # 训练100次把参数复制到explorer  # 400是论文数据
    test_freq = 1000  # 1000次训练后evaluation

    learning = Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done,
              lock, n_training, param_update_freq,
              test_freq, queues, transitions])
    learning.start()
    tasks.append(learning)

    # Add evaluator
    save_model_interval = 1000
    n_evaluation = 3
    evaluating = Process(
        target=evaluator,
        args=[is_training_done, queues[-1], save_model_interval, n_evaluation, episode_max_steps, False])
    evaluating.start()
    tasks.append(evaluating)

    # prefetch = Process(
    #     target=prefetch,
    #     args=[is_training_done, queues[-2], global_rb, lock, trained_steps])
    # prefetch.start()
    # tasks.append(prefetch)

    for task in tasks:
        task.join()
