import numpy as np
from tqdm import tqdm
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement


class SensatUrban:
    def __init__(self):

        # https://github.com/QingyongHu/SensatUrban/blob/master/main_SensatUrban.py

        self.id2name = {0: 'Ground',
                        1: 'High Vegetation',
                        2: 'Buildings',
                        3: 'Walls',
                        4: 'Bridge',
                        5: 'Parking',
                        6: 'Rail',
                        7: 'traffic Roads',
                        8: 'Street Furniture',
                        9: 'Cars',
                        10: 'Footpath',
                        11: 'Bikes',
                        12: 'Water'}

        self.labels_map = dict((v, k) for k, v in self.id2name.items())
        self.num_total_points = 0
        self.semantic_dict = dict(zip(self.labels_map.values(), np.zeros(len(self.labels_map))))

        self.get_stats(root_path="C:\\Users\\Diana\\Desktop\\DATA\\sensaturban\\", pattern="*.ply")

    def get_stats(self, root_path, pattern="\\data_3d_semantics\\*\\*\\static\\*.ply"):
        pcds = glob.glob(f'{root_path}{pattern}')
        num_pcds = len(pcds)
        print(f'Found {num_pcds} PCs in all folders')

        for pcdFile in tqdm(pcds):
            print(pcdFile)
            try:
                pcd_p = PlyData.read(pcdFile)
            except EOFError:
                print(f"Broken file {pcdFile}. Skipping")
                continue
            try:
                data = np.asarray(pcd_p['vertex']['class'])
                self.count_stats(data)
            except ValueError:
                print(f'PCD missing class info {pcdFile}. Skipping!')
                continue
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

        self.plot_barplot(named_semantic_dict, 'sensaturban_all.png')

    def plot_barplot(self, dict, name_to_save="fig.png", top_n=20):
        total_sum = sum(dict.values())
        print(f"TOTAL NUM OF POINTS: {total_sum}")
        print(dict)

        sns.set(rc={'figure.figsize': (15, 7)})
        ax = sns.barplot(x=list(dict.keys())[:top_n], y=[int(i) for i in dict.values()][:top_n])
        percentage = [v / total_sum * 100 for v in dict.values()][:top_n]
        patches = ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height() + .06
            ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title("SENSATURBAN (UAV, areal)")
        plt.savefig(name_to_save, bbox_inches="tight", dpi=600)
        plt.show()


ds = SensatUrban()
