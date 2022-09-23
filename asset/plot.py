import unittest
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt, pylab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from utils.data.partition import barh_report
from utils.result import get_scalar


def plot(df: DataFrame, **kwargs):
    ax = sns.lineplot(data=df, palette="tab10", linewidth=kwargs.get('line_width', 2.5))
    ax.set_xlabel(kwargs.get('x_label'), None)
    ax.set_ylabel(kwargs.get('y_label'), None)
    ax.set_title(kwargs.get('title'), None)
    return ax


def plot_plus(df: DataFrame, **kwargs):
    ax = plot(df, **kwargs)
    ax_ = inset_axes(
        ax, bbox_transform=ax.transAxes,
        width=kwargs.get('width', '25%'),
        height=kwargs.get('height', '25%'),
        loc=kwargs.get('loc', 'upper center'),
        bbox_to_anchor=kwargs.get('bbox_to_anchor', (0, 0, 1, 1)),
    )
    ax_.plot(df, alpha=0.7)
    x = df.index.to_numpy()
    x_left, x_right = int(kwargs.get('x_left', 0.2) * len(x)), int(kwargs.get('x_right', 0.28) * len(x))
    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio, y_ratio = kwargs.get('x_ratio', 0.2), kwargs.get('y_ratio', 0.05)
    xl = x[x_left] - (x[x_right] - x[x_left]) * x_ratio
    xr = x[x_right] + (x[x_right] - x[x_left]) * x_ratio
    # Y轴的显示范围
    y = np.hstack([df[col][x_left:x_right].to_numpy() for col in df.columns])
    yl = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    yu = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    # 调整子坐标系的显示范围
    ax_.set_xlim(xl, xr)
    ax_.set_ylim(yl, yu)
    mark_inset(
        ax, ax_,
        loc1=kwargs.get('loc1', 3),
        loc2=kwargs.get('loc1', 4),
        fc="none", ec="0.5",
        zorder=3
    )
    return ax, ax_


def plot_test(dataset: str, parts: list[str], size: tuple = (24, 12), excluded: list[str] = None):
    plt.rc('font', family='Times New Roman', size='15', weight='normal')
    sns.set_style('whitegrid')
    n = len(parts)
    fig = plt.figure(figsize=size)
    for i, p in enumerate(parts):
        fig.add_subplot(int(f'2{n}{i + 1}'))
        plot(
            get_scalar(f'../benchmark/{dataset}/{p}/output/', 'test_all/acc', 5, excluded),
            line_width=3.5,
            x_label=f'Round\n({chr(97 + i)})',
            y_label=f'Test acc' if i == 0 else None,
            title=p.upper() if p in ['iid'] else p.capitalize().replace('_', ' ')
        )
        fig.add_subplot(int(f'2{n}{i + 1 + n}'))
        plot(
            get_scalar(f'../benchmark/{dataset}/{p}/output/', 'test_all/loss', 5, excluded),
            line_width=3.5,
            x_label=f'Round\n({chr(97 + i + n)})',
            y_label=f'Test loss' if i == 0 else None,
        )
    plt.show()


def plot_mnist_wd(parts: list[str], size: tuple = (16, 12)):
    plt.rc('font', family='Times New Roman', size='15', weight='normal')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=size)
    n = len(parts)
    for i, p in enumerate(parts):
        fig.add_subplot(int(f'2{n}{i + 1}'))
        plot_plus(
            get_scalar(f'../benchmark/src/mnist/{p}/output/', 'diff'),
            title=p.capitalize().replace('_', ' '),
            x_label=f'Round\n({chr(97 + i)})',
            y_label=f'WD' if i == 0 else None,

        )
        fig.add_subplot(int(f'2{n}{i + 1 + n}'))
        plot_plus(
            get_scalar(f'../benchmark/src/mnist/{p}/output/', 'test_all/loss', 5),
            x_label=f'Round\n({chr(97 + i + n)})',
            y_label=f'Test loss' if i == 0 else None,
        )
    plt.show()


def plot_cifar10(parts: list[str], size: tuple = (24, 12), excluded: list[str] = None):
    plt.rc('font', family='Times New Roman', size='15', weight='normal')
    sns.set_style('whitegrid')
    n = len(parts)
    fig = plt.figure(figsize=size)
    for i, p in enumerate(parts):
        fig.add_subplot(int(f'2{n}{i + 1}'))
        plot_plus(
            get_scalar(f'../benchmark/cifar10/{p}/output/', 'test_all/acc', 5, excluded),
            line_width=3.5,
            x_label=f'Round\n({chr(97 + i)})',
            y_label=f'Test acc' if i == 0 else None,
            title=p.upper() if p in ['iid'] else p.capitalize().replace('_', ' '),
            x_left=0.8,
            x_right=0.9
        )
        fig.add_subplot(int(f'2{n}{i + 1 + n}'))
        plot_plus(
            get_scalar(f'../benchmark/cifar10/{p}/output/', 'test_all/loss', 5, excluded),
            line_width=3.5,
            x_label=f'Round\n({chr(97 + i + n)})',
            y_label=f'Test loss' if i == 0 else None,
            x_left=0.8,
            x_right=0.9
        )
    plt.show()


def plot_mnist_partition():
    def plot_(path, tag):
        df = pd.read_csv(path)
        ax = df.iloc[:10, 3:].plot.barh(stacked=True)
        p = path.name.rstrip(".csv")
        ax.set_title(p.upper() if p in ['iid'] else p.capitalize().replace('_', ' '))
        ax.set_ylabel('Client')
        ax.set_xlabel(f'Num')

    plt.rc('font', family='Times New Roman', size='15', weight='normal')
    for i, p in enumerate(['iid', 'class_1', 'dir_0.1']):
        plot_(Path("./mnist").joinpath(f'{p}.csv'), f'({chr(97 + i)})')
        # 调整图片边缘距离
        plt.subplots_adjust(bottom=0.2)
    plt.show()


# data_niid_0_keep_20_train_8
def plot_synthetic_niid_0_20(path: str, excluded: list[str] = None):
    plt.rc('font', family='Times New Roman', size='15', weight='normal')
    sns.set_style('whitegrid')
    plot(
        get_scalar(path, 'test_all/', 5, excluded),
        line_width=3.5,
    )

    plt.show()


class MyTestCase(unittest.TestCase):

    def test_mnist_wd(self):
        plot_mnist_wd(['fixed_interval', 'momentum_beta'])

    def test_mnist(self):
        plot_test('mnist', ['iid', 'class_1', "dir_0.1"])

    def test_cifar10(self):
        plot_cifar10(['new'], (7, 12), excluded=['cfl'])

    def test_plot_data_partition(self):
        plot_mnist_partition()
