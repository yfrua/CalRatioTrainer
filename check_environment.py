import tensorflow as tf

print(tf.config.list_physical_devices())

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")


# try pytorch to find the gpu
import torch 
print(torch.__version__)
print(torch.cuda.is_available())
