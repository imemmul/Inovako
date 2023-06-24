import random
import numpy as np

random.seed(0)

# detection model classes
CLASSES = ('Crack', 'Hole', 'Necking')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

MASK_COLORS = np.array([(255, 56, 56),
                        (255, 178, 29),
                        (146, 204, 23),],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5

