import numpy as np
from torch import nn
import yaml
import matplotlib.pyplot as plt

# LOGGER
class Logger(object):
    def __init__(self, file):
        self.file = file
    def save_log(self, text):
        print(text)
        with open(self.file, 'a') as f:
            f.write(text + '\n')

# PLOT POSE        
def plot_pose(pose):
    print(pose.shape)
    plt.figure()
    plt.scatter(pose[0,:],pose[1,:],color='red')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    for i in range(pose.shape[1]):
        plt.annotate(i, (pose[0,i], pose[1,i]), textcoords='offset points', xytext=(5,-10))
    plt.show()
    
# CONFIG FILE    
def read_yaml(path):
    """
    Read a yaml file from a certain path.
    """
    stream = open(path, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, base, warmup_steps):
        super(CosineSchedule, self).__init__()

        self.base = base.type(torch.FloatTensor)
        self.warmup_steps = warmup_steps.type(torch.FloatTensor)
        self.total_steps = total_steps.type(torch.FloatTensor)

    def __call__(self, step):
        step = step.type(torch.FloatTensor)
        
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        lr = self.base * 0.5 * (1. + torch.cos(np.pi * progress))

        if self.warmup_steps:
            lr = lr * torch.min(1., step / self.warmup_steps)
        return lr

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=20000.0, decay_step=20000.0):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model.type(torch.FloatTensor)
        self.warmup_steps = warmup_steps
        self.decay_step = decay_step
        
    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
        
        return config

    def __call__(self, step):
        step = step.type(torch.FloatTensor)
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return torch.where(step > self.decay_step, lambda: tf.FloatTensor(1e-4),
                       lambda: torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2))

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32) 