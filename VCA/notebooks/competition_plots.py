from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from copy import deepcopy as dc

import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'figure.figsize' : [3.125, 1.93],
                           'legend.fontsize': 8, 'legend.fancybox': True,
                           'font.family': 'serif', 'font.sans-serif': 'Times'})



VCAs = ["meet", "teams", "zoom"]
flows = ["meet", "teams", "zoom", "netflix", "youtube", "iperf"]

speed_vals = [0.5, 1, 2, 3, 4, 5]

colors   = {"zoom" : "#2D8CFF", "meet" : "#00832D", "teams" : "#5451CC", 
            "netflix" : "k", "youtube" : "#E62117", "iperf" : "0.7", "iperfup" : "0.7"}

labels = {"zoom" : "Zoom", "meet" : "Meet", "teams" : "Teams",
          "netflix" : "Netflix", "youtube" : "YouTube", 
          "iperf" : "iPerf3", "iperfup" : "iPerf3",
          "ul" : "Upload", "dl" : "Download"}

webrtc_metrics = {"webrtc_recv_fps" : "FPS"}

figdir = "figs"


def build_dl_dataset():
        
    ### FILES FOR A
    A_data = sorted(glob("/data/jsaxon/vca/ALL-RUNS/DOWNLOAD-A/*/*/*red.csv"))
    a_pcap = pd.concat([pd.read_csv(f) for f in A_data])
    a_vars = ["ts", "shape_dl", "A", "pcap_dl_mbps", "pcap_ul_mbps", "tag"]

    a_pcap_sel = a_pcap[a_vars].rename(columns = {"pcap_dl_mbps" : "a_pcap_dl_mbps", 
                                                  "pcap_ul_mbps" : "a_pcap_ul_mbps", 
                                                  "shape_dl" : "shape"})
    
    a_pcap_sel.sort_values("ts", inplace = True)
    a_pcap_sel.reset_index(inplace = True, drop = True)
    
    ###  FILES FOR B
    B_data = sorted(glob("/data/jsaxon/vca/ALL-RUNS/DOWNLOAD-B/*/*/*red.csv") + 
                    glob("/data/jsaxon/vca/ALL-RUNS/DOWNLOAD-B/*/*/*/*red.csv"))
    b_pcap = pd.concat([pd.read_csv(f) for f in B_data])

    b_vars = ["ts", "B", "pcap_dl_mbps", "pcap_ul_mbps", "active"]
    b_pcap_sel = b_pcap[b_vars].rename(columns = {"pcap_dl_mbps" : "b_pcap_dl_mbps", 
                                                  "pcap_ul_mbps" : "b_pcap_ul_mbps"})
    b_pcap_sel.sort_values("ts", inplace = True)
    b_pcap_sel.reset_index(inplace = True, drop = True)
    
    
    ds = a_pcap_sel.merge(b_pcap_sel, on = "ts", how = "outer")
    ds["active"] = ds["active"].fillna(False)

    tag_vals = ds.query("active")[["A", "B", "shape", "tag"]].sort_values(["A", "B", "shape", "tag"])
    tag_vals = tag_vals.drop_duplicates().reset_index(drop = True)
    tag_vals["R"] = tag_vals.groupby(["A", "B", "shape"]).rank()#.astype(int)
    run_dict = tag_vals.set_index("tag").R.to_dict()
    
    ds["run"] = ds.tag.replace(run_dict)
    ds.loc[ds.run > 3, "run"] = -1
        
    return ds


def build_ul_dataset():
        
    ### FILES FOR A
    A = "/data/jsaxon/vca/ALL-RUNS/UPLOAD/*/*/*A_red.csv"
    a_pcap = pd.concat([pd.read_csv(f) for f in glob(A)])
    a_vars = ["ts", "shape_dl", "A", "pcap_dl_mbps", "pcap_ul_mbps", "tag"]

    a_pcap_sel = a_pcap[a_vars].rename(columns = {"pcap_dl_mbps" : "a_pcap_dl_mbps", 
                                                  "pcap_ul_mbps" : "a_pcap_ul_mbps",
                                                  "shape_dl" : "shape"})
    
    a_pcap_sel.sort_values("ts", inplace = True)
    a_pcap_sel.reset_index(inplace = True, drop = True)
    
    ###  FILES FOR B
    B = "/data/jsaxon/vca/ALL-RUNS/UPLOAD/*/*/*B_red.csv"
    b_pcap = pd.concat([pd.read_csv(f) for f in glob(B)])

    b_vars = ["ts", "B", "pcap_dl_mbps", "pcap_ul_mbps", "active"]
    b_pcap_sel = b_pcap[b_vars].rename(columns = {"pcap_dl_mbps" : "b_pcap_dl_mbps", 
                                                  "pcap_ul_mbps" : "b_pcap_ul_mbps"})
    b_pcap_sel.sort_values("ts", inplace = True)
    b_pcap_sel.reset_index(inplace = True, drop = True)
    
    ds = a_pcap_sel.merge(b_pcap_sel, on = "ts", how = "outer")

    tag_vals = ds[["A", "B", "shape", "tag"]].sort_values(["A", "B", "shape", "tag"])
    tag_vals = tag_vals.drop_duplicates().reset_index(drop = True)
    tag_vals["R"] = tag_vals.groupby(["A", "B", "shape"]).rank()#.astype(int)
    run_dict = tag_vals.set_index("tag").R.to_dict()
    
    ds["run"] = ds.tag.replace(run_dict)
        
    return ds


def plot_bw(A = "teams"):

    f, ax = plt.subplots()

    for dl in speed_vals:
        
        speed_cut = dl_ds.query(f"(A == '{A}') & (B == '{A}') & (shape == {dl})")
        
        if not speed_cut.shape[0]: continue
        speed_cut.a_pcap_dl_mbps\
                 .hist(histtype = "step", alpha = 1, 
                       bins = np.arange(0, 2, 0.04), density = True,
                       label = f"{dl:.1f}", ax = ax)
        
    ax.set_title(labels[A])

    ax.legend()
    

    
def plot_iperf_competition(df, A, ymax = 3, direcs = ["dl"]):

    cmap = plt.get_cmap("viridis")

    f, ax = plt.subplots(len(direcs), 1, figsize = (10, 5 if len(direcs) > 1 else 3.2), 
                         sharex = True, subplot_kw = {"aspect" : 1})

    if len(direcs) == 1: ax = [ax]

    for direc, axi in zip(direcs, ax):

        speeds = df.query(f"(A == '{A}') & (B == 'iperf') & active")

        for bw_i, bw in enumerate(speed_vals):

            c = cmap(bw_i / (len(speed_vals) - 1))
            label = f"{bw:.1f}" if axi == ax[-1] else None

            speeds.query(f"{bw} == shape")\
                  .plot(x = "b_pcap_dl_mbps", y = f"a_pcap_{direc}_mbps", kind = "scatter", 
                        color = c, lw = 0, alpha = 0.25, label = label, ax = axi)

            axi.plot([bw, 0], [0, bw], color = "k", lw = 0.5, ls = ":")

            axi.set_xlim(0, 5)
            axi.set_ylim(0, ymax)
            
            axi.set_xlabel("TCP Flow [Mbps]")
            axi.set_ylabel("VCA Flow [Mbps]")

        if axi == ax[-1]:
            
            anchor = (1, 1)
            if len(direcs) == 1: anchor = (0.88, 0.62)
                
            l = axi.legend(title = "Link\nConstraint\n[Mbps]", 
                           loc = "center left", bbox_to_anchor = anchor)
            
            plt.setp(l.get_title(), multialignment='center')
            for lh in l.legendHandles: lh.set_alpha(1)
                
    ax[0].set_title(labels[A])

    f.savefig(f"{figdir}/{A}_iperf_scatter.pdf")


baselines = {
    "meet" : {"ul" : 0.9, "dl" : 0.8},
    "teams" : {"ul" : 1.3, "dl" : 1.9},
    "zoom" : {"ul" : 0.75, "dl" : 0.96}
}
def get_profiles(ds, A, direction = "dl"):
    
    D = direction
    apps = ["meet", "teams", "zoom"]
    
    if D == "dl":
        apps.extend(["netflix", "youtube", "iperf"])
    else:
        apps.append("iperfup")
            
    profiles = {}
    for B in apps:

        profiles[B] = \
        ds.query(f"(A == '{A}') & (B == '{B}') & active")\
          [[f"shape", f"a_pcap_{D}_mbps", f"b_pcap_{D}_mbps"]].groupby(f"shape")\
          .mean().rename(columns = {f"a_pcap_{D}_mbps" : "VCA", f"b_pcap_{D}_mbps" : "Competing Flow"})

    for prof in profiles.values():

        prof["tot"]          = prof[["VCA", "Competing Flow"]].sum(axis = 1)
        prof["fr_vca"]       = prof["VCA"] / prof["tot"]
        prof["fr_competing"] = prof["Competing Flow"] / prof["tot"]
        prof["fr_vca_max"]   = prof["VCA"] / baselines[A][direction]
        prof["vca_rate"]     = prof["VCA"] 

    return profiles



flow_types = {
    "all" : ["meet", "zoom", "teams", "netflix", "youtube", "iperf", "iperfup"],
    "app" : ["netflix", "youtube", "iperf", "iperfup"],
    "vca" : ["meet", "zoom", "teams"]
}
def bitrate_competition(profiles, tag = "all", D = "dl", VCAs = ["meet", "teams", "zoom"]):

    f, ax = plt.subplots(1, len(VCAs), figsize = (5.5 * len(VCAs), 2.8), sharex = True)
    
    solo = len(VCAs) == 1
    
    if solo: ax = [ax]

    vca_range = list(range(len(VCAs)))
    for vca_i, vca, abc in zip(vca_range, VCAs, "abc"):
        
        for comp in flow_types[tag]:

            if comp not in profiles[vca]: continue

            df = profiles[vca][comp]
            
            axi = ax[vca_i]

            df.vca_rate.plot(label = labels[comp], color = colors[comp], ax = axi)
            
            axi.set_ylim(0, 2.01)
            axi.set_yticks(np.arange(0, 2.1, 0.5))

            axi.set_xlim(0.45, 5.05)
            axi.set_xlabel("Shaped Link [Mbps]")
            axi.set_ylabel("VCA Bitrate [Mbps]")
            
            axi.plot([0, 5], [0, 5], color = "0.7", ls = (0, (8, 8)), lw = 0.5)
            axi.plot([0, 5], [0, 2.5], color = "0.7", ls = (0, (8, 8)), lw = 0.5)
                        
            if vca_i == len(VCAs) - 1:
                axi.legend(title = "Competing\nApplication", 
                           fontsize = 13, loc = "center left", bbox_to_anchor = (1, 0.5))

            if not solo: axi.set_title(f"({abc}) {labels[vca]}")

    txt_args = {"fontsize" : 11, "fontstyle" : "oblique", 
                "color" : "0.3", "backgroundcolor" : "w", 
                "ha" : "center", "va" : "center"}
    ax[0].text(1.75, 1.75, "Capacity", **txt_args)
    ax[0].text(3.50, 1.75, "Half-Capacity", **txt_args)

            
    for axi in ax:
        axi.grid(axis = "y", color = "0.7", lw = 0.5)

    plt.subplots_adjust(hspace = 0.25, wspace = 0.35)
    
    vca_tag = ""
    if solo: vca_tag = VCAs[0] + "_"
   
    f.savefig(f"{figdir}/{D}_competition_{vca_tag}{tag}.pdf")
    # f.savefig(f"{figdir}/{D}_competition_{tag}.png")
        

def plot_time_series(df, A, B, d, cap, run = 3, axi = None):
    
    if not axi: f, axi = plt.subplots(figsize = (6, 3))
    
    ts_series = df.query(f"(A == '{A}') & (B == '{B}') & (shape == {cap}) & active & (run == {run})").ts
    if not ts_series.shape[0]: return
    
    ts_start = ts_series.min()
    ts_min   = ts_start - 30
    ts_max   = ts_start + 150

    ts_ex = df.query("({} < ts) & (ts < {})".format(ts_min, ts_max)).copy()
    ts_ex["dts"] = (ts_ex.ts - ts_start)

    ts_ex.rename(columns = {f"a_pcap_{d}_mbps" : A, f"b_pcap_{d}_mbps" : B + "_"}, inplace = True)
    ts_ex.set_index("dts", inplace = True)

    ts_ex[A].plot(ax = axi, color = colors[A], label = labels[A])
    ts_ex[B + "_"].plot(ax = axi, color = colors[B], label = labels[B])

    axi.set_xlabel("Time Since Initiating Competing Flow [sec]")
    axi.set_ylabel("Bitrate [Mbps]")
    
    # axi.spines["left"].set_visible(False)
    axi.grid(axis = "y", color = "0.7")
    
    return axi

    
def time_series(df, A = "zoom", B = "meet", d = "ul", 
                capacity = speed_vals, ymax = 5):
    
    f, ax = plt.subplots(len(speed_vals), figsize = (8, 12), sharey = True, sharex = True)

    for cap, axi in zip(capacity, ax):

        plot_time_series(df, A, B, d, cap, 1, axi)
        
        if axi == ax[0]: 
            axi.legend(borderpad = 0.4, frameon = True,
                       loc = "upper right", bbox_to_anchor = (0.98, 0.95), )

        axi.set_ylim(0, ymax)

        direction = labels[d]
        axi.set_title(f"{direction} Constrained to {cap} Mbps", y = 0.97)

    plt.subplots_adjust(hspace = 0.4)
    
    f.savefig(f"{figdir}/{A}_{B}_{d}_time_series.pdf")
    # f.savefig(f"{figdir}/{A}_{B}_{d}_time_series.png")


def netflix_time_series(df, cap = 3, runs = [1, 1, 1]):
    
    f, ax = plt.subplots(len(VCAs), figsize = (8, 8), sharey = True, sharex = True)

    for vca, abc, axi, run in zip(VCAs, "abc", ax, runs):

        plot_time_series(df, vca, "netflix", "dl", cap, run, axi)
        axi.set_title(f"({abc}) {labels[vca]}", fontsize = 15)
        
    ax[0].set_ylim(0, 3)
    
    lines, names = [], []
    for vca in VCAs + ["netflix"]:
        names.append(labels[vca])
        lines.append(Line2D([-1], [-1], color = colors[vca], lw = 2))

    leg = ax[0].legend(lines, names, loc = "lower left", 
                       bbox_to_anchor = (0.85, 0.26), 
                       borderpad = 0.4, frameon = True, fontsize = 14)

    plt.subplots_adjust(hspace = 0.4)
    
    f.savefig(f"{figdir}/netflix_time_series.pdf")
    # f.savefig(f"{figdir}/netflix_time_series.png")


def box_plot_single(ds, A = "zoom", shape = 0.5, D = "dl"):

    cut = ds.query(f"active & (A == '{A}') & (shape == {shape})").copy()
    cut["Competing Flow's Share of Link"] = cut[f"b_pcap_{D}_mbps"] / shape

    labels_short = dc(labels)
    if D == "dl": 
        labels_short.pop("iperfup")
        flow_order = ["meet", "teams", "zoom", "netflix", "youtube", "iperf"]
        order = ["Meet", "Teams", "Zoom", "Netflix", "YouTube", "iPerf3"]

    if D == "ul": 
        labels_short.pop("iperf")
        flow_order = ["meet", "teams", "zoom", "iperfup"]
        order = ["Meet", "Teams", "Zoom", "iPerf3"]
        
    cut["B"] = cut.B.replace(labels_short)

    f, ax = plt.subplots(figsize = (6, 4))
    sns.boxplot(data = cut, x = "B", y = "Competing Flow's Share of Link", 
                order = order,
                boxprops = {"zorder" : 10},
                medianprops = {"zorder" : 12, "color" : "0.2"},
                whis = [5, 95],
                ax = ax)

    ax.legend().remove()

    ax.set_xlabel("Competing Flow")
    # ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis = "y", zorder = -10)
    
    ax.set_title("Incumbent Flow: " + labels[A])
    
    patches = [c for c in ax.get_children() if type(c) is matplotlib.patches.PathPatch]
    npatches = len(patches)

    for pi, p in enumerate(patches):

        p.set_facecolor(colors[flow_order[pi]])
        p.set_edgecolor("0.2")

    f.savefig(f"figs/box_plot_{A}_{D}_{shape:.1f}.pdf")
    
    return ax

def box_plot(ds, A = "zoom", B = None, shape = 0.5, D = "dl", tag = "all",
             comp_apps = ["meet", "teams", "zoom", "netflix", "youtube", "iperf"],
             verbose = False):

    if A and B is None: 
        cut = ds.query(f"active & (A == '{A}') & (shape == {shape})").copy()
    elif B and A is None: 
        cut = ds.query(f"active & (B == '{B}') & (shape == {shape})").copy()
    else:
        print("specify either A or B")
        return
    
    labels_short = dc(labels)
    if D == "dl": labels_short.pop("iperfup")
    if D == "ul": labels_short.pop("iperf")
        
    
    order = [labels_short[f] for f in comp_apps]
        
    if B: 
        cut["A"] = cut.A.replace(labels_short)
        cut["B"] = labels_short[B]
    if A: 
        cut["B"] = cut.B.replace(labels_short)
        cut["A"] = labels_short[A]
    
    cutA = cut[["A", "B", f"a_pcap_{D}_mbps"]].copy()
    cutA["Application"] = cutA["A"]
    cutA["Share of Link"] = cutA[f"a_pcap_{D}_mbps"] / shape
    cutA["Incumbent"] = True

    cutB = cut[["A", "B", f"b_pcap_{D}_mbps"]].copy()
    cutB["Application"] = cutB["B"]
    cutB["Share of Link"] = cutB[f"b_pcap_{D}_mbps"] / shape
    cutB["Incumbent"] = False

    reference_flow = labels_short[A if A else B]
    cuts = pd.concat([cutA, cutB], axis = 0)
    
    if A: query = f"(A == '{reference_flow}') & B.isin(@order)"
    if B: query = f"(B == '{reference_flow}') & A.isin(@order)"
    cuts.query(query, inplace = True)
    
    if verbose: 
        print(cuts.groupby(["A", "B", "Incumbent"])["Share of Link"].median().reset_index())

    f, ax = plt.subplots(figsize = (6, 4))
    sns.boxplot(x = "AB"[bool(A)], y = "Share of Link", 
                hue ="Incumbent", data = cuts,
                order = order, hue_order = [True, False], 
                boxprops={"zorder" : 10}, medianprops={"zorder" : 12, "color" : "0.2"},
                whis = [5, 95], # split = True,
                ax = ax)

    if A: 
        ax.set_xlabel("Competing Flow")
    else: 
        ax.set_xlabel("Incumbent Flow")
        
    # ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 1)
    ax.grid(axis = "y", zorder = -10)
    
    if A: ax.set_title("Incumbent Flow: " + labels[A], fontsize = 15)
    if B: ax.set_title("Competing Flow: " + labels[B], fontsize = 15)
    
    ax.legend().remove()
    
    patches = [c for c in ax.get_children() if type(c) is matplotlib.patches.PathPatch]
    # patches = [c for c in ax.get_children() if type(c) is matplotlib.collections.PolyCollection]
    npatches = len(patches)

    flow_order = comp_apps
    for pi, p in enumerate(patches):

        if (pi % 2) == bool(B):
            p.set_facecolor("w")
        else:
            p.set_facecolor(colors[flow_order[pi // 2]])

        p.set_edgecolor("0.2")
        p.set_zorder(2)
        
#     for c in ax.get_children():
#         #if type(c) is matplotlib.patches.Rectangle:
#         c.set_zorder(11)
#         if type(c) is matplotlib.collections.PathCollection:
#             c.set_zorder(12)

    ref_label = A if A else B

    f.savefig(f"figs/box_plot_{ref_label}_{D}_{shape:.1f}_{tag}.pdf")
    
    return ax


