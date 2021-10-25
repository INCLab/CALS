'''
    Set the distance(threshold) and two targets.
    Then, find number of frames(times) that two targets are closer than the distance.
'''
import sys
import pandas as pd

threshold = 2
target_id1 = 1001
target_id2 = 1002

result = pd.read_csv('tmp/global_result.txt', delimiter=' ', header=None)
result.columns = ['frame', 'id', 'x', 'y']

print(result)


