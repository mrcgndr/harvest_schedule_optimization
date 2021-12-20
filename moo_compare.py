import click
import util
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_selection, get_reference_directions
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.moead import MOEAD
from pymoo.optimize import minimize
import pickle
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option("-si", "--site", "site", type=int, help="site number (0 or 1)", prompt="Site")
@click.option("-sc", "--scenario", "scenario", type=int, help="scenario number (1, 2, or 3)", prompt="Scenario")
@click.option("-c", "--cyclic", "cyclic_years", default=False, is_flag=True, help="consider cyclic years", prompt="cyclic years")
@click.option("-a", "--algorithm", "algo", help="used algorithm (nsga2 or moead)", prompt="Algorithm")
@click.option("-n", "--n-eval", "n_eval", type=int, default=1000000, help="number of evaluations")
def main(site, scenario, cyclic_years, algo, n_eval):
    o = util.HarvestOptimization(data_dir = "data/")
    o.loadSiteScenario(site, scenario, cyclic_years)

    if algo == "nsga2":
        EA_moo = NSGA2( pop_size = 1,
                        sampling=util.MySampling(site, scenario, cyclic_years),
                        mutation=get_mutation("int_pm"),
                        crossover=get_crossover("int_one_point"))
        EA_hlo = NSGA2( pop_size = 1,
                        sampling=util.MySampling(site, scenario, cyclic_years),
                        mutation=get_mutation("int_pm"),
                        crossover=get_crossover("int_one_point"))
    elif algo == "moead":
        EA_moo = MOEAD( get_reference_directions("das-dennis", n_dim = 6, n_partitions=2),
                        seed=1,
                        sampling=util.MySampling(site, scenario, cyclic_years),
                        mutation=get_mutation("int_pm"),
                        crossover=get_crossover("int_one_point"))
        EA_hlo = MOEAD( get_reference_directions("das-dennis", n_dim = 2, n_partitions=2),
                        seed=1,
                        sampling=util.MySampling(site, scenario, cyclic_years),
                        mutation=get_mutation("int_pm"),
                        crossover=get_crossover("int_one_point"))
    else:
        raise ValueError("Only 'nsga2' and 'moead' are allowed as algorithms.")

    p_moo = util.HarvestOptimizationMOO(site, scenario, cyclic_years)
    p_hlo = util.HarvestOptimizationHLO(site, scenario, cyclic_years)

    res_moo = minimize( p_moo,
                        EA_moo,
                        termination=('n_eval', n_eval),
                        verbose=True,
                        seed=1,
                        save_history=False)

    with open(f"results/moo_n_iter{n_eval}_site{site}_scenario{scenario}_{algo}_moo.pkl", "wb") as f:
        pickle.dump(res_moo, f)

    del res_moo

    res_hlo = minimize( p_hlo,
                        EA_hlo,
                        termination=('n_eval', n_eval),
                        verbose=True,
                        seed=1,
                        save_history=False)

    with open(f"results/moo_n_iter{n_eval}_site{site}_scenario{scenario}_{algo}_hlo.pkl", "wb") as f:
        pickle.dump(res_hlo, f)

    del res_hlo

    o = util.HarvestOptimization("data/")

    o.loadSiteScenario(site = site, scenario = scenario, cyclic_years = cyclic_years)

    print("Harvest schedule optimization ...")
    fit = o.optimize(n_eval, daily=False, plot_path="")

    np.save(f"results/optimize_n_iter{n_eval}_site{site}_scenario{scenario}_nrand0"+("_cyclic" if cyclic_years else "")+".npy", fit)

    res = []
    names = []

    with open(f"results/moo_n_iter{n_eval}_site{site}_scenario{scenario}_{algo}_moo.pkl", "rb") as f:
        res.append(pickle.load(f))
        names.append("6-obj. NSGA-II" if algo=="nsga2" else "6-obj. MOEA/D")
    with open(f"results/moo_n_iter{n_eval}_site{site}_scenario{scenario}_{algo}_hlo.pkl", "rb") as f:
        res.append(pickle.load(f))
        names.append("2-obj. NSGA-II" if algo=="nsga2" else "2-obj. MOEA/D")
    
    res.append(np.load(f"results/optimize_n_iter{n_eval}_site{site}_scenario{scenario}_nrand0"+("_cyclic" if cyclic_years else "")+".npy", allow_pickle=True))
    names.append("hierarchical (1+1)-ES")

    whs = []

    for r in res:
        try:
            whs.append(o.getHarvest(r.X[0])[0])
        except:
            whs.append(o.getHarvest(r[1][0])[0])
            
    figwidth, figheight = 5.12, 3.84

    fig, ax = plt.subplots()
    ax.set(title="site 1, scenario 1", xlabel="harvest week", ylabel="distance to optimum")

    if o.capacity_limit:
        cap = o.capacity_limit

    for wh, name in zip(whs, names):

        max_perc_hi  = 100*(np.max(wh/cap)-1)
        max_capacity = np.max(wh)
        abs_max_diff = np.max(np.abs(wh[wh>0]-cap))
        abs_med_diff = np.median(np.abs(wh[wh>0]-cap))
        n_weeks      = np.sum(wh > 0)
        n_over_cap   = np.sum(wh > cap)
        hl           = util.calc_hl(wh, cap)
        info_str =(#f"$OR=$ {max_perc_hi:.2f} %\n"
                    #f"$\Delta_{{max}}=$ {abs_max_diff:0g}\n"
                    #f"$\Delta_{{median}}=$ {abs_med_diff:.1f}\n"
                    #f"$C_{{need}}=$ {max_capacity:0g}\n"
                    #f"$n_h=$ {n_weeks:0g}\n"
                    #f"$n_{{>C}}=$ {n_over_cap:0g}\n"
                    f"$\\vec{{L}}=$ ({hl[0]:.2f}, {hl[1]:.2f})")

        wh_dist = np.min(np.dstack((np.abs(wh-6000), wh)), axis=2)[0]

        ax.bar(np.arange(len(wh))[wh>0], wh_dist[wh>0], alpha=0.8 if name=="hierarchical (1+1)-ES" else 0.4, label=f"{name}\n"+info_str)

    ax.legend(ncol=1, loc=0)
    ax.grid()
    fig.set_size_inches(figwidth*1.5, figheight, forward=True)
    fig.savefig(f"plots/comparison_multiobjective_hierarchical_{algo}.pdf", bbox_inches="tight", dpi=250)


if __name__ == "__main__":

    main()