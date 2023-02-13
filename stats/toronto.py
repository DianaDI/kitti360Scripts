import numpy as np
from tqdm import tqdm
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from plyfile import PlyData


class TorontoDS:
    def __init__(self):

        """

            Road (label 1)
            Road marking (label 2)
            Natural (label 3)
            Building (label 4)
            Utility line (label 5)
            Pole (label 6)
            Car (label 7)
            Fence (label 8)
            unclassified (label 0)

        """

        self.labels_map = {"unclassified": 0,
                           "Road": 1,
                           "Road markings": 2,
                           "Natural": 3,
                           "Building": 4,
                           "Utility line": 5,
                           "Pole": 6,
                           "Car": 7,
                           "Fence": 8}

        self.id2name = dict((v, k) for k, v in self.labels_map.items())
        self.num_total_points = 0
        self.semantic_dict = dict(zip(self.labels_map.values(), np.zeros(len(self.labels_map))))

        self.get_stats(root_path="C:\\Users\\Diana\\Desktop\\DATA\\Toronto_3D\\Toronto_3D\\", pattern="*.ply")

    def get_stats(self, root_path, pattern="\\data_3d_semantics\\*\\*\\static\\*.ply"):
        pcds = glob.glob(f'{root_path}{pattern}')
        num_pcds = len(pcds)
        print(f'Found {num_pcds} PCs in all folders')

        for pcdFile in tqdm(pcds):
            pcd_p = PlyData.read(pcdFile)
            data = np.asarray(pcd_p['vertex']['scalar_Label'])
            self.count_stats(data)
        self.show_stats()

    def count_stats(self, pcd_data):
        self.num_total_points += len(pcd_data)
        for i in pcd_data:
            self.semantic_dict[i] += 1

    def show_stats(self):
        print(f"TOTAL NUM OF POINTS--: {self.num_total_points}")
        print(self.semantic_dict)

        semantic_dict = dict(sorted(self.semantic_dict.items(), key=lambda item: item[1], reverse=True))
        named_semantic_dict = {}
        for i in semantic_dict:
            named_semantic_dict[self.id2name[i]] = semantic_dict[i]

        # todo dump calculated dict
        self.plot_barplot(named_semantic_dict, './toronto_all.png')

    def plot_barplot(self, dict, name_to_save="fig.png", top_n=20):
        total_sum = sum(dict.values())
        print(f"TOTAL NUM OF POINTS: {total_sum}")
        print(dict)

        sns.set(rc={'figure.figsize': (9, 5)})
        ax = sns.barplot(x=list(dict.keys())[:top_n], y=[int(i) for i in dict.values()][:top_n])
        percentage = [v / total_sum * 100 for v in dict.values()][:top_n]
        patches = ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height() + .06
            ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')

        plt.xticks(rotation=90)
        plt.ylabel("Number of points")
        plt.xlabel("Object classes")
        plt.tight_layout()
        plt.title("TORONTO3D")
        plt.savefig(name_to_save, bbox_inches="tight", dpi=300)
        plt.show()


ds = TorontoDS()
