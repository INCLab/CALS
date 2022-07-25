import cv2

cap1 = cv2.VideoCapture('0218_part1.mp4')
cap2 = cv2.VideoCapture('0218_part2.mp4')

# frame1 = int(round(cap1.get(cv2.CAP_PROP_FPS),0))
# frame2 = int(round(cap2.get(cv2.CAP_PROP_FPS),0))
frame1 =10
frame2 = 24

fram_list1 = [1]*frame1
fram_list2 = [1]*frame2

#


def findone(list):
    count  = []
    for i, value in enumerate(list):
        if value == 1:
            count.append(i)
    return count

def syncVideo(list1, list2):
    list_one1 = list1.count(1)
    list_one2 = list2.count(1)
    print(list_one1, list_one2)
    if list_one1 == list_one2:
        return list1, list2
    else:
        dif = abs(list_one1 - list_one2)
        one_loc1 = findone(list1)
        one_loc2 = findone(list2)
        if dif == 1:
            if list_one1 > list_one2:
                list1[one_loc1[-1]] = 0
            else:
                list2[one_loc2[-1]] = 0
        else:
            i = max(list_one1, list_one2) // dif
            if i == 1:
                i += 1

            if list_one1 > list_one2:
                for j in range(i - 1, len(one_loc1), i):
                    list1[one_loc1[j]] = 0
            else:
                for j in range(i -1, len(one_loc2), i):
                    list2[one_loc2[j]] = 0
    print(list1)
    print(list2)
    return syncVideo(list1, list2)

syncVideo(fram_list1, fram_list2)

