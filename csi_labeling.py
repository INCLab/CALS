'''
    Usage: python csi_labeling.py labelclass testname
'''
import os
import time
import argparse

from loguru import logger
from datetime import datetime
from labeling import personExistLabeling, gridAllocateLabeling


def make_parser():
    parser = argparse.ArgumentParser("CSI Labeling")

    parser.add_argument("label", default="PE", help="Person Existence(PE,pe,Pe) or Grid Allocation(GA,ga,Ga)")
    parser.add_argument("test_name", default=None, help="the name of test folder")
    return parser


args = make_parser().parse_args()

label = None

if args.label in ['PE', 'Pe', 'pe']:
    label = 'PE'
    logger.info("Start Person Existence labeling!")
elif args.label in ['GA', 'ga', 'Ga']:
    label = 'GA'
    logger.info("Start Grid Allocation labeling!")
else:
    logger.info("Wrong label argument!")
    exit()

if args.test_name not in os.listdir('data'):
    logger.info("Wrong test name argument!")
    exit()

# ========= Read Data =========
test_name = args.test_name
csi_path = os.path.join('data', test_name, 'csi')
mot_path = os.path.join('data', test_name, 'mot')

# ========= Output data path ========
out_path = None
if label == 'PE':
    out_path = os.path.join('data', test_name, 'labeled', 'PE')
else:
    out_path = os.path.join('data', test_name, 'labeled', 'GA')

os.makedirs(out_path, exist_ok=True)

# =========  Create file list  =========
csi_flist = os.listdir(csi_path)
mot_flist = os.listdir(mot_path)

if label == 'PE':
    personExistLabeling(mot_flist, csi_flist, mot_path, csi_path, out_path)
    logger.info("Done")
elif label == 'GA':
    gridAllocateLabeling(mot_flist, csi_flist, mot_path, csi_path, out_path)
    logger.info("Done")
