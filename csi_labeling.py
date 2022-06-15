'''
    Usage: python csi_labeling.py label_class testname
'''
import os
import time
import argparse

from loguru import logger
from datetime import datetime
from labeling import personExistLabeling, gridAllocateLabeling, noPersonLabeling


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_parser():
    parser = argparse.ArgumentParser("CSI Labeling")

    parser.add_argument("label", default="PE", help="Person Existence(PE,pe,Pe) Grid Allocation(GA,ga,Ga)")
    parser.add_argument("test_name", default=None, help="the name of test folder")
    parser.add_argument("--np", type=str2bool, default=False, help="All data class is no person in PE process")
    parser.add_argument("--pnp", type=str2bool, default=False, help="Part of data class is no person in PE process")
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

test_nameList = args.test_name.split('/')
if test_nameList[0] not in os.listdir('data'):
    logger.info("Wrong test name argument!")
    exit()

if len(test_nameList) > 1:
    if test_nameList[1] not in os.listdir('data/' + test_nameList[0]):
        logger.info("Wrong test name2 argument!")
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

if label == 'PE' and args.np is False:

    if args.pnp is True:
        time_ms_list = [
            '2022-06-08 18:47:00',  # Start
            '2022-06-08 18:52:00',  # End
        ]
        timeList = []
        for t in time_ms_list:
            timeList.append(time.mktime(datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timetuple()))

        personExistLabeling(mot_path, csi_path, out_path, timeList)
    else:
        personExistLabeling(mot_path, csi_path, out_path)
    logger.info("Done")
elif label == 'PE' and args.np is True:
    time_ms_list = [
        '2022-06-15 23:03:00',  # Start
        '2022-06-15 23:05:00',  # End
    ]
    timeList = []
    for t in time_ms_list:
        timeList.append(time.mktime(datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timetuple()))
    noPersonLabeling(timeList, csi_path, out_path)
    logger.info("Done")
elif label == 'GA':
    gridAllocateLabeling(mot_path, csi_path, out_path)
    logger.info("Done")
