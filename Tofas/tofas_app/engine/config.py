import random
import numpy as np

random.seed(0)

# detection model classes
CLASSES = ('Crack', 'Hole', 'Necking')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 4) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5

