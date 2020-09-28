import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    model = pickle.load(open("../../Data/Ongoing/Model/RF_model.sav", 'rb'))


    plt.rcParams["figure.figsize"] = 10,4

    x_p = np.linspace(-3,3, num=len(model.feature_importances_))
    y_p = model.feature_importances_

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [x_p[0]-(x_p[1]-x_p[0])/2., x_p[-1]+(x_p[1]-x_p[0])/2.,0,1]
    ax.imshow(y_p[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    ax2.plot(x_p,y_p)

    plt.tight_layout()
    plt.show()
    plt.savefig("../../Data/Ongoing/Figures/Feature_importance.png")

    