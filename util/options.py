import numpy as np


def RandShuffle(config):
    train_size = config.train_size

    if config.scenes == 'all':
        if config.db_name == 'WIN5-LID':
            scenes = list(range(220))
        elif config.db_name == 'win5all':
            scenes = list(range(17820))
        elif config.db_name == 'win5_5×5':
            scenes = list(range(5500))
        elif config.db_name == 'win5_random':
            scenes = list(range(220*config.sel_num))
        elif config.db_name == 'win5_81':
            scenes = list(range(220))
        elif config.db_name == 'win5_9×9':
            scenes = list(range(1980))

    else:
        scenes = config.scenes

    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))  # np.floor: 返回不大于输入参数的最大整数。（向下取整）
    n_test_scenes = n_scenes - n_train_scenes

    seed = np.random.random()
    random_seed = int(seed * 10)
    np.random.seed(random_seed)  # 生成指定随机数，仅一次有效--把np.random的随机数给指定了
    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_train_scenes:]

    return train_scene_list, test_scene_list
