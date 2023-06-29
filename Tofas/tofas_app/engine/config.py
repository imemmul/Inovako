import random
import numpy as np

random.seed(0)

# detection model classes
CLASSES = ('Cracking', 'Hole', 'Necking')

# colors for per classes


MASK_COLORS = np.array([(0, 0, 255), (0, 255, 0), (255, 0, 0)], dtype=np.float32) / 255.


COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}
# alpha for segment masks
ALPHA = 0.5

