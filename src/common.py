import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from datetime import timedelta
import calendar
import glob
from multiprocessing import Pool
import json
import collections
from IPython.display import display
import re
import os
import inspect
import importlib
import pickle
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.ticker import ScalarFormatter

matplotlib.style.use('ggplot')
font = {'family': 'Noto Sans CJK JP'}
matplotlib.rc('font', **font)
# %config InlineBackend.figure_format = "retina"
# %matplotlib inline
fontsize = 20
dpi = 400
quality=100
plt.rcParams["font.size"] = fontsize

pd.options.display.max_rows = 800
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 500
pd.options.display.width = 500

# helper function

def set_display_style():
    """
    general display setting for jupyter notebook
    
    """
    
    matplotlib.style.use('ggplot')
    font = {'family': 'Noto Sans CJK JP'}
    matplotlib.rc('font', **font)
    plt.rcParams["font.size"] = 20
    
    pd.options.display.max_rows = 800
    pd.options.display.max_columns = 500
    pd.options.display.max_colwidth = 500
    pd.options.display.width = 500


def describes(data):
    """
    show DataFrame summary
    
    
    data: pd.DataFrame
    """
    assert type(data) == pd.DataFrame, "must be pd.DataFrame"
    
    info = pd.DataFrame({"type": pd.Series(data.dtypes)})
    info["records"] = [data[column].dropna().shape[0] for column in data.columns]
    info["uniques"] = info.index.map(lambda x: len(data[x].dropna().unique()))
    info["uniqueness"] = (info["uniques"]/info["records"]).map(lambda x: round(100*x,1))
    info["completeness [%]"] = info.index.map(lambda x: round(100 * (data[x].dropna().shape[0] / data.shape[0]), 1))
    info["5-examples"] = info.index.map(lambda x: str(data[x].dropna().unique()[0:5].tolist()))
    info = info.sort_values(by=["completeness [%]", "uniqueness"], ascending=[False, False])
    
    return info 
    
    
def show_timestamp(msg=""):
    """
    show current timestamp for jupyter notebook
    
    msg: additional display message
    """
    
    
    func = inspect.stack()[1].function
    ntime = datetime.now().isoformat()
    display("{}: {}: {}".format(ntime, func, msg))


# for datetime processing

_weekday_pattern = ["月", "火", "水", "木", "金", "土", "日"]

def get_weekday(value):
    if pd.isnull(value):
        return "Nat"
    return _weekday_pattern[value.weekday()]

def date_to_str(value):
    if pd.isnull(value):
        return "Nat"
    return "{} ({})".format(value.strftime("%Y/%m/%d"), get_weekday(value))

def date_to_month_str(value):
    if pd.isnull(value):
        return "Nat"
    return value.strftime("%Y/%m")

def get_week_num_interval(dt):
    n = (dt.weekday() + 1) % 7
    begin = dt - timedelta(days=n)
    end = dt + timedelta(days=6-n)
    return (begin, end)

def date_to_week_str(value):
    if pd.isnull(value):
        return "Nat"
    tmp = get_week_num_interval(value)
    return "第" + value.strftime("%U") + f"週（{tmp[0].strftime('%m/%d')}～{tmp[1].strftime('%m/%d')}）"

def timedelta_to_hms(td):
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

# for plotting

class FigCtx:
    def __init__(self, output_dir=None):
        """
        
        output_dir: output directory of saved figures. if None, does not always save figures
        """
        
        self.output_dir = output_dir
        if self.output_dir is not None:
            if os.path.exists(self.output_dir):
                if not os.path.isdir(self.output_dir):
                    raise ValueError("already exists, but is not directory: " + self.output_dir)
            else:
                os.mkdir(self.output_dir)
            
        
    def _save_and_show(self, title, no_save=False):
        if not no_save and self.output_dir is not None:
            plt.savefig("{}/{}.png".format(self.output_dir, title), dpi=dpi, bbox_inches="tight")
        plt.show()
    
    def _fmt_title(self, title, desc):
        display(title)
        t = "{}\n\n{}".format(title, desc)
        return t
    
    @staticmethod
    def _check_layout(data, layout):
        ncol = len(data.columns.values)
        nl = layout[0] * layout[1]
        if nl >= ncol:
            return "single"
        if len(data.columns.names) > 1: # multi index
            if nl >= len(data.columns.get_level_values(0).unique()):
                return "multi"
        assert False, "layout={}, but columns={}".format(layout, data.columns.values)
        
    @staticmethod
    def _min_max(data):
        v_max = data.max()
        if type(v_max) is pd.Series:
            v_max = max(v_max)
        
        v_min = data.min()
        if type(v_min) is pd.Series:
            v_min = min(v_min)
        return (v_min, v_max) 
    

    def bar(self, result, title, desc, stacked=True, figsize=(20,20), no_save=True, use_math_text=False, layout=None, share=True, outer_legend=False):
        t = self._fmt_title(title, desc)
        if layout is None:
            plt.figure()
            ax = result.plot(title=t, kind="bar", cmap="jet", stacked=stacked, figsize=figsize)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=use_math_text))
            if outer_legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)   
        else:
            ret = self._check_layout(result, layout)
            if ret == "single":
                result.plot(title=t, kind="bar", cmap="jet", stacked=stacked, figsize=figsize, subplots=True, layout=layout, sharex=True, sharey=share)
                if outer_legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  
            else:
                fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize, sharex=True, sharey=share)
                plt.suptitle(t, fontsize=fontsize)
                display(axes)
                level0_cols = result.columns.get_level_values(0).unique().tolist()
                for i in range(0, len(level0_cols)):
                    tmp = result.T[result.columns.get_level_values(0) == level0_cols[i]].T
                    tmp.plot(kind="bar", cmap="jet", stacked=stacked, ax=axes[i])   
                    if outer_legend:
                        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0) 
        self._save_and_show(title, no_save)
        
        
    def line(self, result, title, desc, figsize=(20,20), no_save=True, use_math_text=False, layout=None, share=True, outer_legend=False, xrotate=True):
        t = self._fmt_title(title, desc)
        if layout is None:
            plt.figure()
            ax = result.plot(title=t, kind="line", cmap="jet", figsize=figsize)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=use_math_text))
            if outer_legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            if xrotate:
                plt.xticks(rotation=90)
        else:
            ret = self._check_layout(result, layout)
            if ret == "single":
                result.plot(title=t, kind="line", cmap="jet", figsize=figsize, subplots=True, layout=layout, sharex=True, sharey=share)
                if outer_legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  
            else:
                fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize, sharex=True, sharey=share)
                plt.suptitle(t, fontsize=fontsize)
                display(axes)
                level0_cols = result.columns.get_level_values(0).unique().tolist()
                for i in range(0, len(level0_cols)):
                    tmp = result.T[result.columns.get_level_values(0) == level0_cols[i]].T
                    tmp.plot(kind="line", cmap="jet", ax=axes[i])
                    if outer_legend:
                        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0) 
        self._save_and_show(title, no_save)
    
    def bar_line(self, bar, line, title, desc, stacked=True, figsize=(20,20), no_save=True, use_math_text=False, outer_legend=False, bar_label=None, line_label=None, adjust_ylim=False):
        t = self._fmt_title(title, desc)
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        
        if bar_label is None:
            if type(bar) is pd.Series:
                ax1.set_ylabel(bar.name)
        else:
            ax1.set_ylabel(bar_label)
        
        if line_label is None:
            if type(line) is pd.Series:
                ax2.set_ylabel(line.name)
        else:
            ax2.set_ylabel(line_label)
        
        ax1_range = self._min_max(bar)
        ax2_range = self._min_max(line)
        
        if adjust_ylim:
            ax1_m = max([abs(ax1_range[0]), abs(ax1_range[1])])
            ax2_m = max([abs(ax2_range[0]), abs(ax2_range[1])])
            ax1_range = (-ax1_m, ax1_m)
            ax2_range = (-ax2_m, ax2_m)
        
        bar.plot(title=t, kind="bar", cmap="jet", ylim=ax1_range, stacked=stacked, ax=ax1)    
        line.plot(kind="line", ylim=ax2_range, ax=ax2)
        
        if outer_legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)       
        self._save_and_show(title, no_save) 
    
    
    def box(self, x, y, labels, title, desc, figsize=(20,20), no_save=True, sym="", hue=None, hue_order=None, order=None, xrotate=True):
        t = self._fmt_title(title, desc)
        plt.figure(figsize=figsize)
        plt.title(t, fontsize=fontsize, loc="center")
        if labels is not None and order is None:
            order = labels
        ax = sns.boxplot(x=x, y=y, order=order, sym=sym)
        if xrotate:
            ax.set_xticklabels(rotation=90, labels=labels)
        self._save_and_show(title, no_save)     

    def violin(self, x, y, labels, title, desc, figsize=(20,20), no_save=True, sym="", hue=None, hue_order=None, order=None, xrotate=True):
        t = self._fmt_title(title, desc)
        plt.figure(figsize=figsize)
        plt.title(t, fontsize=fontsize, loc="center")
        if labels is not None and order is None:
            order = labels
        ax = sns.violinplot(x=x, y=y, order=order, sym=sym)
        if xrotate:
            ax.set_xticklabels(rotation=90, labels=labels)
        self._save_and_show(title, no_save)    
        
    def heatmap(self, result, title, desc, figsize=None, no_save=True, annot=True, fmt="2.2f"):
        t = self._fmt_title(title, desc)
        if figsize is None:
            figsize = (result.shape[1] * 2.0, result.shape[0] * 1.0)
        plt.figure(figsize=figsize)
        plt.title(t, fontsize=fontsize, loc="center")
        sns.heatmap(result, cbar=False, annot=annot, cmap="jet", fmt=fmt)
        self._save_and_show(title, no_save)
        
    def pie(self, result, title, desc, figsize=(10,10), no_save=True):
        labels = result.index.tolist()
        t = self._fmt_title(title, desc)
        plt.figure(figsize=figsize)
        plt.title(t, fontsize=fontsize, loc="center")
        plt.pie(result, autopct="%1.2f%%", counterclock=False, startangle=90)
        plt.legend(labels, bbox_to_anchor=(1, 1.02,))
        self._save_and_show(title, no_save)
        
    def graph(self, graph, title, desc, figsize, no_save=True):
        at = self._fmt_title(title, desc)
        plt.figure(figsize=figsize)
        G = nx.DiGraph()
        for s, t, w in zip(graph["src"], graph["dest"], graph["weight"]): 
            G.add_edge(s, t, weight=w) 
        pos = nx.shell_layout(G)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={ (i, j): w['weight'] for i, j, w in G.edges(data=True) }, label_pos=0.3, font_size=fontsize)
        nx.draw(G, pos, node_size=3000, node_color='#CC5', width=3, edge_color="#AAA", connectionstyle="arc,rad=0.1")
        nx.draw_networkx_labels(G, pos, font_size=fontsize, font_family='Noto Sans CJK JP')
        plt.title(at, loc="left", fontsize=fontsize)
        self._save_and_show(title, no_save)
