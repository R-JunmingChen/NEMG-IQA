import json
import random
from collections import OrderedDict

# Define DB information
TRAIN_RATIO = 0.8
datasets=["live","csiq","tid2013"]
num_scene={"csiq":30,"live":29,"tid2013":25}
origin_scores_path={"csiq":"origin_score/CSIQ.txt","live":"origin_score/LIVE_IQA.txt","tid2013":"origin_score/TID2013.txt"}



def make_score_files(num=10):
    for dataset in datasets:
        ALL_SCENES = list(range(num_scene[dataset]))
        for i in range(num):
            make_score_file(origin_scores_path[dataset],"{dataset_name}_{index}.json".format(dataset_name=dataset,index=str(i)),ALL_SCENES)

def make_score_file(origin_score_file,score_file,all_scenes):
    """
        Make train and test image list from database
    """
    scenes, dist_types, d_img_list, r_img_list, score_list = [], [], [], [], []
    with open(origin_score_file, 'r') as listFile:
        for line in listFile:
            (scn_idx, dis_idx, ref, dis, score) = line.split()
            scenes.append(int(scn_idx))
            dist_types.append(int(dis_idx))
            r_img_list.append(ref)
            d_img_list.append(dis)
            score_list.append(float(score))

    n_images = len(r_img_list)

    # divide scene randomly get train and test datasetScores
    random.shuffle(all_scenes)
    train_scenes_indexs = all_scenes[:int(TRAIN_RATIO * len(all_scenes))]
    test_scenes_indexs = all_scenes[int(TRAIN_RATIO * len(all_scenes)):len(all_scenes)]

    train_scenes, train_dist_types, train_d_img_list, train_r_img_list, train_score_list = [], [], [], [], []
    test_scenes, test_dist_types, test_d_img_list, test_r_img_list, test_score_list = [], [], [], [], []
    for index in range(n_images):
        if scenes[index] in train_scenes_indexs:  # train
            train_scenes.append(scenes[index])
            train_dist_types.append(dist_types[index])
            train_d_img_list.append(d_img_list[index])
            train_r_img_list.append(r_img_list[index])
            train_score_list.append(score_list[index])
        else:
            test_scenes.append(scenes[index])  # test
            test_dist_types.append(dist_types[index])
            test_d_img_list.append(d_img_list[index])
            test_r_img_list.append(r_img_list[index])
            test_score_list.append(score_list[index])

    ret = OrderedDict()
    ret['train'] = OrderedDict()
    ret['test'] = OrderedDict()

    ret['train']['dis'] = train_d_img_list
    ret['train']['ref'] = train_r_img_list
    ret['train']['mos'] = train_score_list
    ret['train']['type'] = train_dist_types

    ret['test']['dis'] = test_d_img_list
    ret['test']['ref'] = test_r_img_list
    ret['test']['mos'] = test_score_list
    ret['test']['type'] = test_dist_types


    with open(score_file, 'w+') as f:
        json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    make_score_files(10)
