import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kitti360scripts.helpers.labels import id2name


# kitti360 all, semantic kitti, sensaturban
all_distributions_map = {"Vegetation": [28.8, 27.1, 25.1],
                         "Road": [23.3, 19.5, 20.3],
                         "Building": [10.6, 12.9, 39.8],
                         "Sidewalk": [8.1, 14, 2],
                         "Ground (inc. grass, terrain, other ground)": [4.7 + 1.3, 8.7 + 0.3, 20],  # grass, terrain, ground
                         "Car (inc. car, truck, bus, trailer)": [3.1 + 0.2 + 0.1, 4.6 + 0.3 + 0.2, 1.7],
                         # kitti: include car, truck, trailer; semanitic kitti: car, truck, bus
                         "Parking": [2, 1.4, 2.3],
                         "Wall (inc. fence)": [2 + 0.8, 6.4, 1],  # kitti: wall, fence; semantic kitti: fence;
                         "Street furniture (inc. pole and lights)": [0.2, 0.3, 1.2]}  # kitti: pole

avg_distr_map = dict((k, np.mean(v)) for k, v in all_distributions_map.items())
avg_distr_map = dict(sorted(avg_distr_map.items(), key=lambda item: item[1], reverse=True))


semkitti = {'vegetation': 773132761.0, 'road': 555049215.0, 'sidewalk': 398464216.0, 'building': 368681416.0, 'terrain': 247714395.0, 'fence': 182644338.0, 'car': 130287591.0, 'unlabeled': 95909070.0, 'parking': 40525481.0, 'trunk': 19692693.0, 'other-ground': 9631626.0, 'pole': 8378712.0, 'bus': 7678156.0, 'truck': 5094794.0, 'traffic-sign': 1823430.0, 'person': 1293759.0, 'motorcycle': 1281899.0, 'bicycle': 639562.0, 'bicyclist': 605464.0, 'motorcyclist': 110516.0}
kitti360 = {'vegetation': 265166536.0, 'road': 213822769.0, 'building': 97802750.0, 'sidewalk': 74405909.0, 'terrain': 43060916.0, 'car': 28810389.0, 'parking': 18623885.0, 'wall': 18294040.0, 'ground': 11649563.0, 'garage': 11463167.0, 'fence': 7649530.0, 'rail track': 3194970.0, 'gate': 2058912.0, 'truck': 1876428.0, 'pole': 1477914.0, 'trailer': 835100.0, 'trash bin': 703872.0, 'box': 676432.0, 'unknown object': 616520.0, 'guard rail': 598628.0, 'traffic sign': 462966.0, 'unknown construction': 368669.0, 'unknown vehicle': 313371.0, 'caravan': 303450.0, 'stop': 247842.0, 'tunnel': 187902.0, 'person': 174849.0, 'motorcycle': 152512.0, 'smallpole': 122002.0, 'bicycle': 113476.0, 'lamp': 104012.0, 'train': 103523.0, 'bus': 66174.0, 'traffic light': 21720.0, 'vending machine': 15574.0, 'bridge': 5723.0, 'rider': 865.0, 'unlabeled': 0.0, 'ego vehicle': 0.0, 'rectification border': 0.0, 'out of roi': 0.0, 'static': 0.0, 'dynamic': 0.0, 'polegroup': 0.0, 'sky': 0.0, 'license plate': 0.0}


# {0: 27010402.0, 1: 86796722.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 11649563.0, 7: 213822769.0, 8: 74405909.0, 9: 18623885.0, 10: 3194970.0, 11: 97802750.0, 12: 18294040.0, 13: 7649530.0, 14: 598628.0, 15: 5723.0, 16: 187902.0, 17: 1477914.0, 18: 0.0, 19: 21720.0, 20: 462966.0, 21: 265166536.0, 22: 43060916.0, 23: 0.0, 24: 174849.0, 25: 865.0, 26: 28810389.0, 27: 1876428.0, 28: 66174.0, 29: 303450.0, 30: 835100.0, 31: 103523.0, 32: 152512.0, 33: 113476.0, 34: 11463167.0, 35: 2058912.0, 36: 247842.0, 37: 122002.0, 38: 104012.0, 39: 703872.0, 40: 15574.0, 41: 676432.0, 42: 368669.0, 43: 313371.0, 44: 616520.0, -1: 0.0}
# kitti360_named = {}
# for i in kitti360:
#     kitti360_named[id2name[i]] = kitti360[i]
# print(kitti360_named)

print("bla")


def intersect(a, b):
    print(f'Length of 1: {len(a)}')
    print(f'Length of 2: {len(b)}')
    intersection = set(a).intersection(set(b))
    print(f'Length of intersection: {len(intersection)}')
    print(intersection)
    return intersection


def plot_barplot(dict, name_to_save="fig.png"):
    total_sum = sum(dict.values())
    print(f"TOTAL: {total_sum}")
    print(dict)

    sns.set(rc={'figure.figsize': (20, 10)})
    ax = sns.barplot(x=list(dict.keys()), y=[int(i) for i in dict.values()])
    percentage = list(dict.values())
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width() / 2
        y = patches[i].get_height() + .06
        ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(name_to_save, bbox_inches="tight", dpi=600)
    plt.show()


def get_percentage(dictionary):
    dict1_sum = sum(list(dictionary.values()))
    dict1_perc = {k: v / dict1_sum * 100 for (k, v) in dictionary.items()}
    return dict1_perc


def plot_kittis(cols, dicts):
    df = pd.DataFrame(columns=["label", "percent", "dataset"])

    names = {0: "kitti360", 1: "semantic_kitti"}

    for i in range(len(dicts)):
        d = dicts[i]
        d_perc = get_percentage(d)

        d_itersectiion = {k: val for (k, val) in d_perc.items() if k in cols}
        intersection_perc  = sum(list(d_itersectiion.values()))
        print(f'TOTAL % SUM: {intersection_perc}')

        for col in cols:
            df = df.append({'dataset': f'{names[i]} ({round(intersection_perc)}%)', "label": col, "percent": d_perc[col]}, ignore_index=True)
    df = df.sort_values(by=['percent'], ascending=False)
    g = sns.catplot(
        data=df, kind="bar",
        x="label", y="percent", hue="dataset",
        ci="sd", palette="mako", alpha=.6, height=6
    )
    # g.despine(left=True)
    g.set_axis_labels("Common labels", "Percentage")
    g.legend.set_title("")
    #plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Semantic labels distribution over points for 2 datasets")
    plt.savefig("kittis.png", bbox_inches="tight", dpi=600)
    plt.show()

# plot_barplot(avg_distr_map, "averaged.png")
common_labels = intersect(list(semkitti.keys()), list(kitti360.keys()))
plot_kittis(common_labels, [kitti360, semkitti])


