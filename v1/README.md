## Introduction
Version 1. Got max return after 15 minutes' training

### How to run
Python apex.py

### Details
- Environment："PongDeterministic-v4" in gym
- Algorithm：apex
- Framework：tensorflow2

### Files
- "results" folder：Saved training data. Use "tensorboard --logdir results" to visualize
- "model" folder: Saved training model. Use "model=tf.keras.models.load_model
  ('xx', custom_objects={'tf': tf})" to restore
- agent.py: Apex agent
- apex.py: Apex algorithm, such as multiprocessing

