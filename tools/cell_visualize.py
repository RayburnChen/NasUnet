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
        genotype = Genotype(down=[('down_dil_conv', 0), ('down_dil_conv', 1), ('down_dep_conv', 1), ('dil_conv', 2), ('down_cweight', 1), ('dil_conv', 3)], down_concat=range(2, 5), up=[('up_dep_conv', 1), ('dil_conv', 0), ('up_cweight', 1), ('conv', 2), ('identity', 3), ('up_cweight', 1)], up_concat=range(2, 5))

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
