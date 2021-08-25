import cv2

#


def findone(list):
    count = []

    for i, value in enumerate(list):

        if value == 1:
            count.append(i)

    return count



def syncVideo(list):
    # # list1 = list[0]
    # # list2 = list[1]
    # list_one = []
    #
    # list_one2 = list[1].count(1)
    # for i in range(len(list)):
    #     list_one.append(list[1].count(1))
    # exit()
    #
    # for i in range(len(list))
    #     if list_one[i] == list_one[j]:
    #
    #         return list[0], list[1]
    #
    # else:
    #
    #     dif = abs(list_one1 - list_one2)
    #
    #     one_loc1 = findone(list[0])
    #
    #     one_loc2 = findone(list[1])
    #
    #     if dif == 1:
    #
    #         if list_one1 > list_one2:
    #
    #             list[0][one_loc1[-1]] = 0
    #
    #         else:
    #
    #             list[1][one_loc2[-1]] = 0
    #
    #     else:
    #
    #         i = max(list_one1, list_one2) // dif
    #
    #         if i == 1:
    #             i += 1
    #
    #         if list_one1 > list_one2:
    #
    #             for j in range(i - 1, len(one_loc1), i):
    #                 list[0][one_loc1[j]] = 0
    #
    #         else:
    #
    #             for j in range(i - 1, len(one_loc2), i):
    #                 list[1][one_loc2[j]] = 0

    return syncVideo(list)