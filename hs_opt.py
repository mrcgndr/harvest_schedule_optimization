import os
from util import HarvestOptimization
import util
import numpy as np
import click
from GPy.models import GPRegression
from GPy.kern import RBF, Linear, Bias, PeriodicExponential, White
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import os

figwidth, figheight = 5.12, 3.84
opt_dir = "results/"
plot_dir = "plots/"

#mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = figwidth, figheight
mpl.rcParams['figure.autolayout'] = True

def loadOptimize(n_iter, site, scenario, cyclic):
    glob_path = opt_dir+f"/optimize_n_iter{n_iter}_site{site}_scenario{scenario}_nrand0"+("_cyclic" if cyclic else "")+".npy"
    files = sorted(glob(glob_path))
    optimize = []
    for f in files:
        optimize.append(np.load(f, allow_pickle=True))
    return optimize

def hplots_without_legend(site, scenario, cyclic, n_iter, n_random):
    sc_title = {0: "3", 1: "1", 2: "2-1", 3: "2-2"}
    loss_title = {1: f"$L_+$", 2: f"$L_-$", 3: "loss 3"}

    o = loadOptimize(n_iter, site, scenario, cyclic)

    mut_prob = np.array([np.asarray(o[i][0], dtype=object)[:,3] for i in range(len(o))], dtype=object)
    loss = np.array([np.asarray(o[i][0], dtype=object)[:,2] for i in range(len(o))], dtype=object)
    time = np.array([np.asarray(o[i][0], dtype=object)[:,1] for i in range(len(o))], dtype=object)
    iteration = np.array([np.asarray(o[i][0], dtype=object)[:,0] for i in range(len(o))], dtype=object)
    planting_days = np.array(np.array([np.asarray(o[i][1])[0] for i in range(len(o))]))
    min_loss = np.array([l[-1] for l in loss])

    o = HarvestOptimization("data")
    o.loadSiteScenario(site = site, scenario = scenario, cyclic_years = cyclic)

    wh_orig = o.getHarvest(o.orig_days)[0]
    
    for i in range(len(planting_days)):
        o.loss = min_loss[i]
        o.best_planting_days = planting_days[i]
        wh_opti = o.getHarvest(o.best_planting_days)[0]
        fig, ax = o.plotHarvest(show_original=True, n_random=n_random)

        if o.capacity_limit:
            C = o.capacity_limit
        else:
            if o.scenario == 2:
                C = np.sum(wh_opti)/np.sum(wh_opti>0)
            elif o.scenario == 3:
                C = np.sum(wh_opti)/57.

        overshoot_red  = 1-(np.sum(wh_opti[wh_opti > C] - C)/np.sum(wh_orig[wh_orig > C] - C))
        undershoot_red = 1-(np.sum(C - wh_opti[(wh_opti < C) & (wh_opti > 0)])/np.sum(C - wh_orig[(wh_orig < C) & (wh_orig > 0)]))

        fig.set_size_inches(figwidth*1.2, figheight*1.2, forward=True)
        ax.get_legend().remove()
        #ax.legend(ncol=1, loc="upper right", prop={'size':'small'})
        ax.set(title=f"site {site}, scenario {sc_title[scenario]}\n$R_o$={100*overshoot_red:.02f} %, $R_u$={100*undershoot_red:.02f} %")
        if not cyclic:
            ax.set(xlim=(10,75), ylim=(0,2*C))
            
        fig.savefig(plot_dir+f"/harvest_site{site}_sc{scenario}"+("_cyclic" if cyclic else "")+f".pdf", bbox_inches="tight", dpi=250)

    del loss, time, iteration

def gdu_prediction(o, site):
    X = o.tr_2["day_id"].to_numpy()
    X = X/365.25
    Y = o.tr_2[f"site_{site}"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    noise = White(input_dim=1)
    lin_trend = Linear(input_dim=1)
    bias = Bias(input_dim=1)
    per = PeriodicExponential(input_dim=1, period=1.)
    per.period.constrain_bounded(0.99, 1.01)
    kernel = per + lin_trend + bias + noise

    model = GPRegression(X_train.reshape(-1,1), Y_train.reshape(-1,1), kernel=kernel, normalizer=None)

    model.optimize_restarts(10, parallel=True)

    X_fc = np.arange(3000)/365.25
    Y_fc, Y_fc_std = model.predict(X_fc.reshape(-1,1)) 

    np.save(f"GDU_pred_site{site}.npy", (Y_fc.flatten(), Y_fc_std.flatten()))


@click.command()
@click.option("-si", "--site", "site", type=int, help="site number (0 or 1)", prompt="Site")
@click.option("-sc", "--scenario", "scenario", type=int, help="scenario number (1 for S1, 2 for S2-1, or 3 for S2-2)", prompt="Scenario")
@click.option("-c", "--cyclic", "cyclic_years", default=False, is_flag=True, help="consider cyclic years", prompt="cyclic years")
@click.option("-ni", "--n-iter", "n_iter", type=int, default=1000000, help="number of iterations")
@click.option("-nr", "--n-random", "n_random", type=int, default=10, help="number of random harvest matrix (for plotting yield uncertainty)")
def main(site, scenario, cyclic_years, n_iter, n_random):
    if (not os.path.exists("GDU_pred_site0.npy")) or (not os.path.exists("GDU_pred_site1.npy")):
        o = util.HarvestOptimization("data/")
        print("GDU forecast for site 0 ...")
        gdu_prediction(o, 0)
        print("GDU forecast for site 1 ...")
        gdu_prediction(o, 1)

    o = util.HarvestOptimization("data/")

    o.loadSiteScenario(site = site, scenario = scenario, cyclic_years = cyclic_years)

    print("Harvest schedule optimization ...")
    fit = o.optimize(n_iter, daily=False, plot_path="")

    np.save(f"results/optimize_n_iter{n_iter}_site{site}_scenario{scenario}_nrand0"+("_cyclic" if cyclic_years else "")+".npy", fit)
    print("Make plot ...")
    hplots_without_legend(site, scenario, cyclic_years, n_iter, n_random)

    print("Done.")


if __name__ == "__main__":

    main()