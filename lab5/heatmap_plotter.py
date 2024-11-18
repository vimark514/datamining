import matplotlib.pyplot as plt
import numpy as np


class Heatmap2DPlotter:
    AMOUNT_OF_TICKS_ON_COLORBAR = 5
    COMMA_DIGITS = 2

    def __init__(self, data_2d, title="", xlabel="", ylabel=""):
        self.data_2d = data_2d
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.__setup_plot()
        self.__setup_colorbar()

    def show(self):
        plt.show()

    def __setup_plot(self):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.imshow(self.data_2d, cmap='viridis', interpolation='nearest')

    def __setup_colorbar(self):
        plt.colorbar()
