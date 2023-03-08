import json

""" configuration json """


class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


config = Config({
        # dataset path                  # /win5_all/===/win5/===/MPI/===/NBU/===/win5_epi/

        "db_name":                      "win5",
        # "db_path":                      "/home/lin/Dateset/win5/distorted_images/",
        "db_path": r"D:\000_dataset\SAI\WIN5-LID\HEVC\0",
        "epi_path":                     "/home/lin/Dateset/"+"win5"+"_epi/distorted_images/",
        "text_path":                    "/home/lin/Dateset/win5/1.txt",

        "svPath":                       "./result",
        "mos_sv_path":                  "./data",

        "model_path":                  "./model/resnet50.pth",


        # optimization
        "batch_size":                   4,                               # batch size
        "n_epoch":                      600,                             # epoch
        "val_freq":                     1,                               # 几次训练测试一次
        "crop_size":                    224,                             # dataloader裁切patch大小,224，256, 和img_size一起改动

        # 经常改动
        "aug_num":                      1,                               # patch数
        "sel_num":                      7,                               # 选择sai, 9*9,7*7,5*5,3*3,1*1
        "if_avg":                       False,                            # 测试是否取均值，和aug_num一起使用时, 只增加测试集
        "avg_num":                      1,                               # test每_次平均,epi平均9！！

        # 偶尔改动
        "train_rate":                   0.8,
        "normal_test":                  False,                           # 正常测试 or five_point_crop, True\False
        "if_resize":                    True,                            # True\False

        # 很少改动
        "learning_rate":                1e-5,                            # 学习率， 1e-5, 1e-4
        "weight_decay":                 1e-5,                               # 1e-5, 0
        "T_max":                        50,                             # 50, 3e4
        "eta_min": 0,
        "num_avg_val": 5,
        "num_workers": 0,                                                 # num_workers

        # model
        "input_size":                   (512, 512),                      # (512, 512),(434, 625)
        "patch_size":                   16,                              # 主网络裁切小patch大小，  8,16
        "img_size":                     224,                             # 主网络裁切patch大小, 和crop_size一起改动

        "embed_dim":                    768,                             # 768, 192, 384, 1024
        "dim_mlp":                      768,                             # 768, ……
        "num_heads":                    [4, 4],                          # [4, 4]，2组。 [8, 8]最佳
        "window_size":                  2,                               # swin, window_size, 4,2,和patchsize一起改
        "depths":                       [2, 2],                          # encoder数， [2, 2],2个
        "num_outputs":                  1,
        "num_tab":                      2,                               # attention block数量， 2
        "scale":                        0.13,                            # 0.13

        # optimization & training parameters
        'lr_rate': 1e-4,
        'momentum': 0.9,

        # # ViT structure
        # 'n_enc_seq':                    16*16,       # 20 * 14,
        # 'n_layer': 14,
        # 'd_hidn': 384,
        # 'i_pad': 0,
        # 'd_ff': 384,
        # 'd_MLP_head': 1152,
        # 'attn_head': 6,
        # 'd_head': 384,
        # 'dropout': 0.1,
        # 'emb_dropout': 0.1,
        # 'ln_eps': 1e-12,
        # 'n_output': 1,

        # load & save checkpoint
        "model_name": "model_maniqa",
        "output_path": "./output",
        "snap_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })
