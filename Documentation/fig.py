import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset:
height = [6474, 269]
bars = ('Failed', 'Succes')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

plt.title('Imbalance in growing vs non growing')
plt.ylabel('number of datapoints')


# Show graphic
plt.show()
