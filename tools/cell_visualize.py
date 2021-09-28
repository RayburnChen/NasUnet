import datetime
import sys
import os
import platform

from util import visualize
from util.genotype import Genotype


def main(format):
    if 'Windows' in platform.platform():
        os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
    try:
        genotype = Genotype(down=[('dil_conv_3', 0), ('se_conv', 1), ('dil_conv_3', 0), ('se_conv', 1), ('conv', 2), ('dil_conv_2', 0)], down_concat=range(2, 5), up=[('dil_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 0), ('dil_conv_3', 2), ('dil_conv_3', 3), ('max_pool', 2)], up_concat=range(2, 5), gamma=[1, 0, 0, 1, 1, 1])


    except AttributeError:
        print('{} is not specified in genotype.py'.format(genotype))
        sys.exit(1)

    down_cell_name = '{}-{}'.format("DownC", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    up_cell_name = '{}-{}'.format("UpC", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    visualize.plot(genotype.down, down_cell_name, format=format, directory="./cell_visualize")
    visualize.plot(genotype.up, up_cell_name, format=format, directory="./cell_visualize")


if __name__ == '__main__':
    # support {'jpeg', 'png', 'pdf', 'tiff', 'svg', 'bmp', 'tif', 'tiff'}
    main(format='pdf')
