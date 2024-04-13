# action
import math
from skimage import io, color
import numpy as np
from tqdm import trange

"""算法步骤
1.撒种子。将K个超像素中心分布到图像的像素点上。
2.微调种子的位置。以K为中心的3×3范围内，移动超像素中心到这9个点中梯度最小的点上。这样是为了避免超像素点落到噪点或者边界上。
3.初始化数据。取一个数组label保存每一个像素点属于哪个超像素。dis数组保存像素点到它属于的那个超像素中心的距离。
4.对每一个超像素中心x，它2S范围内的点：如果点到超像素中心x的距离（5维）小于这个点到它原来属于的超像素中心的距离，那么说明这个点属于超像素x。更新dis，更新label。
5.对每一个超像素中心，重新计算它的位置。
重复4 5 两步。"""
class Cluster(object):#定义一个超像素类
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)#它包含聚类的中心点坐标（h, w）和Lab颜色空间中的L, a, b值
        self.pixels = []
        self.no = self.cluster_index#聚类的编号
        self.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)#RGB转换为Lab
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):#创建一个新的聚类
        return Cluster(h, w,
                       self.data[h][w][0],#对应第三通道的L，a，b
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):#初始化聚类中心

        self.K = K
        self.M = M
        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}#label保存每一个像素点属于哪个超像素
        self.dis = np.full((self.image_height, self.image_width), np.inf)#dis数组保存像素点到它属于的那个超像素中心的距离

    def init_clusters(self):
        h = int(self.S / 2)#规则分布像素中心点，每个超像素点的大小为S，两个像素点之间的距离为S
        w = int(self.S / 2)
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = int(self.S / 2)
            h += int(self.S)

    # def get_gradient(self, h, w):#计算图像的梯度
    #     if w + 1 >= self.image_width:#计算有没有超边界，防止访问到图像外的像素点
    #         w = self.image_width - 2
    #     if h + 1 >= self.image_height:
    #         h = self.image_height - 2
    #
    #     gradient = self.data[w + 1][h + 1][0] - self.data[w][h][0] + \
    #                self.data[w + 1][h + 1][1] - self.data[w][h][1] + \
    #                self.data[w + 1][h + 1][2] - self.data[w][h][2]梯度计算有问题，应该水平方向和垂直方向分别计算
    #     return gradient
    def get_gradient(self, h, w):#修改后的梯度计算，不仅计算H,W通道的梯度，还计算了L通道的
        # 初始化梯度的三个分量
        l_gradient = 0
        a_gradient = 0
        b_gradient = 0

        # 计算L通道的梯度分量
        if w < self.image_width - 1:
            l_gradient += self.data[h][w + 1][0] - self.data[h][w][0]
        if w > 0:
            l_gradient -= self.data[h][w - 1][0] - self.data[h][w][0]

        if h < self.image_height - 1:
            l_gradient += self.data[h + 1][w][0] - self.data[h][w][0]
        if h > 0:
            l_gradient -= self.data[h - 1][w][0] - self.data[h][w][0]

            # 计算水平方向的梯度
            horizontal_gradient = 0
            if w < self.image_width - 1:
                horizontal_gradient = self.data[h][w + 1][0] - self.data[h][w][0]
            if w > 0:
                horizontal_gradient -= self.data[h][w - 1][0] - self.data[h][w][0]

            # 计算垂直方向的梯度
            vertical_gradient = 0
            if h < self.image_height - 1:
                vertical_gradient = self.data[h + 1][w][0] - self.data[h][w][0]
            if h > 0:
                vertical_gradient -= self.data[h - 1][w][0] - self.data[h][w][0]

        # 合并三个通道的梯度
        gradient = math.sqrt(l_gradient ** 2 + a_gradient ** 2 + b_gradient ** 2)+horizontal_gradient+vertical_gradient
        return gradient

    def move_clusters(self):#移动聚类中心
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):#便利3*3的中心区域范围，寻找最小的梯度点，更新聚类中心
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):#将像素分配最近的聚类点
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))#距离公式
                    if D < self.dis[h][w]:#更新像素的聚类分配
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):#更新聚类中心的位置
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            name = 'E:\RGB\Lei_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)


if __name__ == '__main__':
    p = SLICProcessor("E:\RGB\Lei.png", 500, 30)
    p.iterate_10times()