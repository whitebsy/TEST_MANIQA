import random

def dis_list3(text_path, train_rate=0.8):
    txt = open(text_path, 'r')
    txt_r = txt.readlines()
    list_all = []
    for line in txt_r:
        idx, _, _ = line.split()
        idx = int(idx)
        list_all.append(idx)

    random.shuffle(list_all)
    train_dis = list_all[0:int(round(train_rate * len(list_all)))]
    test_dis = list_all[int(round(train_rate * len(list_all))):]

    for i in range(len(test_dis)):
        if test_dis[i] in train_dis:
            print('scene error')
            exit()

    return train_dis, test_dis
