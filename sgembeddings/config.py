DATA_DIR = r"D:\data\cora\cora"

# Number of walks to take per node
NUM_WALKS = 1
# How long each walk should be
WALK_LENGTH = 5


BATCH_SIZE = 50
EPOCHS = 4
# Length of number of samples is the number of layers/iterations in each graphsage encoder.
NUM_SAMPLES = [10, 5]
LAYER_SIZES = [50, 50]      
DROPOUT = 0.0

assert(len(LAYER_SIZES) == len(NUM_SAMPLES))
assert(0.0 <= DROPOUT <= 1)
