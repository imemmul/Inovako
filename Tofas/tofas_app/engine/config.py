import random
import numpy as np

random.seed(0)

# detection model classes
CLASSES = ('Crack', 'Hole', 'Necking')

# colors for per classes


MASK_COLORS = np.array([(255, 50, 50), (50, 255, 50), (50, 50, 255)], dtype=np.float32) / 255.


COLORS = {
    'Crack': (255, 50, 50),
    'Hole': (50, 255, 50),
    'Necking': (50, 50, 255)
}
# alpha for segment masks
ALPHA = 0.5

