import numpy as np

def object_side():
    prob_obj_on_left= 0.5
    prob_block_coherence = 0.5
    block_left = np.random.choice([0,1], p=[0.5, 0.5])

    if block_left==0:
        object_on_left=np.random.choice([0,1], p=[prob_block_coherence, 1-prob_block_coherence])
    else:
        object_on_left=np.random.choice([0,1], p=[1-prob_block_coherence, prob_block_coherence])

    return object_on_left

exp = np.zeros(100)
for i in range(len(exp)):
    exp[i] = object_side()

print(np.mean(exp))