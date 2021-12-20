import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.mutation import Mutation


class HarvestOptimization(object):
    
    def __init__(self, data_dir):
        
        self.weekday0 = datetime(2020,1,1).isoweekday()

        self.tr_1 = pd.read_csv(f"{data_dir}/Dataset_1.csv")

        self.tr_2 = pd.read_csv(f"{data_dir}/Dataset_2.csv")

        try:
            gp_site0, gp_site0_std = np.load("GDU_pred_site0.npy")
            gp_site1, gp_site1_std = np.load("GDU_pred_site1.npy")
        
            self.daily = pd.DataFrame()
            
            self.daily["year_day"]    = np.arange(len(gp_site0))
            self.daily["site_0_mean"] = gp_site0
            self.daily["site_0_std"]  = gp_site0_std
            self.daily["site_1_mean"] = gp_site1
            self.daily["site_1_std"]  = gp_site1_std
        except:
            pass
            
    def plotGDUPerDay(self):
        fig, ax = plt.subplots()
        for i in range(2):
            ax.plot(self.daily["year_day"], self.daily[f"site_{i}_mean"], c=f"C{i}")
            ax.fill_between(self.daily["year_day"], self.daily[f"site_{i}_mean"]-self.daily[f"site_{i}_std"], self.daily[f"site_{i}_mean"]+self.daily[f"site_{i}_std"], alpha=.2, label=f"site {i}")
        ax.legend()
        ax.set(xlabel="day of year", ylabel="accumulated GDUs per day")
        ax.grid()
        
        return fig, ax   
    
    def _makeHarvestMatrix(self):
        # H[0,i,j] = harvest day, if plant i is planted on day j
        self.H = -np.ones((1, len(self.orig_days), 700), dtype=int)
        days2week = (np.arange(700)+self.weekday0)//7
        for i in range(len(self.orig_days)):
            acc_gdus = self.cum_gdus[None] - self.cum_gdus[np.arange(self.planting_bounds[i,0]+1, self.planting_bounds[i,1]+2), None]
            harvest_days  = np.array([np.searchsorted(a, self.req_gdus[i])-1 for a in acc_gdus])
            self.H[0,i,self.planting_bounds[i,0]:self.planting_bounds[i,1]+1] = np.array([np.searchsorted(a, self.req_gdus[i])-1 for a in acc_gdus])

        self.H = self.H[:,:,:self.H.max()+1]
        self.d_max = self.H.max()
 
        return self.H
    
    def getRandomHarvestMatrices(self, n_random):
        gdus_rand = np.array([np.random.normal(loc   = self.daily[f"site_{self.site}_mean"].values, 
                                               scale = self.daily[f"site_{self.site}_std"].values) for i in range(n_random)])
        cum_gdus = np.array([np.hstack((0, np.cumsum(g))) for g in gdus_rand])
        # H[n,i,j] = harvest day, if plant i is planted on day j with GDU constellation n
        H = -np.ones((n_random, len(self.orig_days), 700), dtype=int)
        for i in range(len(self.orig_days)):
            acc_gdus = np.array([cum_gdus[j][None] - cum_gdus[j][np.arange(self.planting_bounds[i,0]+1,
                                                                           self.planting_bounds[i,1]+2),None]
                                                                 for j in range(n_random)])
            H[:,i,self.planting_bounds[i,0]:self.planting_bounds[i,1]+1] = np.array([[np.searchsorted(a, self.req_gdus[i])-1 for a in acc] for acc in acc_gdus])
        H = H[:,:,:H.max()+1]
  
        return H
    
    def _getMaxWeeklyHarvest(self):
        H_week = (self.H+self.weekday0)//7
        if self.cyclic_year:
            self.max_weekly_harvest = np.zeros(52)
            for i in range(len(H_week)):
                for j in np.unique(H_week[i])[1:]:
                    self.max_weekly_harvest[j%52] += self.harvest_quant[i]
        else:
            self.max_weekly_harvest = np.zeros(H_week.max()+1)
            for i in range(len(H_week)):
                for j in np.unique(H_week[i])[1:]:
                    self.max_weekly_harvest[j] += self.harvest_quant[i]
                    
        return self.max_weekly_harvest
    
    def _makeWeekMatrix(self):
        # W[i,j] = binary matrix for day i belonging to week j
        # W_year[i,j] = binary matrix for day i belonging to week j considering a cyclic year
        
        days2week = (np.arange(700)+self.weekday0)//7

        if self.cyclic_years:
            self.W = np.zeros((52, len(days2week)))
            for i in range(len(self.W)):
                self.W[i,days2week%52==i] = 1    
        else: 
            self.W = np.zeros((days2week.max()+1, len(days2week)))
            for i in range(len(self.W)):
                self.W[i,days2week==i] = 1

        return self.W 
  
    def loadSiteScenario(self, site, scenario, cyclic_years=False):
        assert site in [0,1]
        assert scenario in [0,1,2,3]
        
        self.site = site
        self.scenario = scenario
        self.cyclic_years = cyclic_years
        
        if scenario <= 1:
            self.capacity_limit = 7000 if site == 0 else 6000
        else:
            self.capacity_limit = None
        
        self.orig_days = self.tr_1["original_planting_day"].values[self.tr_1["site"].values==site]
        
        earliest_days = self.tr_1["early_planting_day"].values[self.tr_1["site"].values==site]
        latest_days = self.tr_1["late_planting_day"].values[self.tr_1["site"].values==site]
        #earliest_days = np.zeros_like(earliest_days_)
        #latest_days = 300*np.ones_like(latest_days_)
        
        self.planting_bounds = np.stack([earliest_days, latest_days]).T
        self.planting_ranges = np.diff(self.planting_bounds).flatten()

        self.req_gdus = self.tr_1["required_gdus"].values[self.tr_1["site"].values==site]
        s = 1 if scenario <= 1 else 2
        self.harvest_quant = self.tr_1[f"scenario_{s}_harvest_quantity"].values[self.tr_1["site"].values==site]
        self.cum_gdus = np.hstack((0, np.cumsum(self.daily[f"site_{site}_mean"].values)))
        
        self.plant_idx = np.arange(len(self.orig_days))
        
        self._makeWeekMatrix()
        self._makeHarvestMatrix()
 
    def getRandomHarvest(self, H, planting_days, daily=False):
        if daily:
            wh = np.sum(np.eye(H.max()+1)[H[:,self.plant_idx,planting_days]]*self.harvest_quant[:,None], axis=1)
        else:
            wh = np.sum(self.W.T[H[:,self.plant_idx,planting_days]]*self.harvest_quant[:,None], axis=1)
        return wh        
    
    def getHarvest(self, planting_days, daily=False):
        if daily:
            wh = np.sum(np.eye(self.d_max+1)[self.H[:,self.plant_idx,planting_days]]*self.harvest_quant[:,None], axis=1)
        else:
            wh = np.sum(self.W.T[self.H[:,self.plant_idx,planting_days]]*self.harvest_quant[:,None], axis=1)
        return wh
     
    def getRandomPlantingDays(self):
        np.random.seed()
        return np.random.randint(*self.planting_bounds.T)
    
    def conc_conv(self, h, capacity):
        h_scaled = h/capacity
        loss = np.vstack(( np.sum(np.where(h_scaled >1, np.exp(h_scaled)-np.e, 0), axis=1),
                           np.sum(np.where(h_scaled<=1, h_scaled*(1-h_scaled), 0), axis=1))).T            
        return loss
    
    def getLoss(self, planting_days, daily):
        h = self.getHarvest(planting_days, daily)
        if self.scenario == 0:
            loss = [[ np.max(np.abs(h[h>0]-self.capacity_limit)),
                      np.median(np.abs(h[h>0]-self.capacity_limit)),
                      np.sum(h>0) ]]
            return loss
        elif self.scenario == 1:
            cap = self.capacity_limit/7. if daily else self.capacity_limit
            return self.conc_conv(h, cap)
        
        elif self.scenario == 2:
            mask = h>0
            return self.conc_conv(h, np.sum(h)/np.sum(mask))
        
        elif self.scenario == 3:
            #return [[np.sum(np.exp(1+h/10000)-np.e)]]
            if self.cyclic_years:
                return self.conc_conv(h, np.sum(h)/52.)
            else:
                return self.conc_conv(h, np.sum(h)/57.)
     
    def mutateVector(self, V, mutation_prob, allow_no_mutation=False):
        np.random.seed()
        if not allow_no_mutation:
            while True:
                change = np.random.random(len(self.planting_ranges)) < mutation_prob
                if np.sum(change) > 0:
                    break
        else:
            change = np.random.random(len(self.planting_ranges)) < mutation_prob

        V_mutated  = V.copy()
        V_mutated[change] = np.array([np.random.randint(low, up+1) for low, up in self.planting_bounds[change]], dtype=int)

        return V_mutated
            
    def compare(self, old_loss, old, new, daily):
        new_loss = self.getLoss(new, daily)
        comp = np.lexsort(np.stack((new_loss,old_loss)).T[::-1])[0] # 0 = new, 1 = old
        
        return [(1+(old_loss != new_loss), new, new_loss), (0, old, old_loss)][comp] 
    
    def compareMulti(self, old_losses, old, new, daily):
        new_losses = self.getLoss(new, daily)
        comps = np.lexsort(np.stack((new_losses, old_losses)).T[::-1])[:,0] # 0 = new, 1 = old
        
        n_old = np.sum(comps)

        return [(1+(np.all(old_losses != new_losses)), new, new_losses), (0, old, old_losses)][len(comps)-n_old < n_old]           
    
    def optimize(self, n_iter, daily, Pstart=[], iter_offset=0, loss_history=[], plot_path=""):
        if len(Pstart) > 0:
            self.best_planting_days = Pstart
        else:
            self.best_planting_days = self.getRandomPlantingDays()
        if len(loss_history) > 0:
            self.loss_history = loss_history
            t_offset = loss_history[-1][1]
        else:
            self.loss_history = []
            t_offset = 0
        self.loss = self.getLoss(self.best_planting_days, daily)
        info_str = f"Site {self.site}, Scenario {self.scenario}, Cyclic {self.cyclic_years}, Daily {daily}"
        t = tqdm(np.arange(iter_offset, iter_offset+n_iter), desc=info_str)
        self.best_solutions = [self.best_planting_days]
        t_start = time.time()
        j = 0
        N = len(self.plant_idx)
        for i in t:
            mut_prob = (1 + (0.01*N-1)*np.sin(0.0005*j)**2)/N
            P_ = self.mutateVector(self.best_planting_days, mutation_prob=mut_prob)

            change, self.best_planting_days, self.loss = self.compareMulti(self.loss, self.best_planting_days, P_, daily)
            if change == 0:
                j += 1
            #elif change == 1:
            #    j = 0
            elif change == 2:
                if (plot_path != "") and (i > 100000):
                    fig, ax = self.plotHarvest(show_original=False, daily=daily)
                    ax.set(title=f"{i} iterations\nloss: {np.mean(self.loss, axis=0)} +/- {np.std(self.loss, axis=0, ddof=1)}, mut. = {(1+np.log10(1+j)):.2f}")
                    fig.savefig(plot_path, bbox_inches="tight")
                    plt.close(fig)
                    del fig, ax
                j = 0
                self.loss_history.append((i, time.time()-t_start+t_offset, self.loss, mut_prob*N))
                self.best_solutions = [self.best_planting_days]
                t.set_postfix_str(f"L = {self.loss}, mut. = {mut_prob*100:.4f} %")
                       
            
        self.loss_history.append((iter_offset+n_iter-1, time.time()-t_start+t_offset, self.loss, mut_prob*N))
            
        return self.loss_history, self.best_solutions
    
    def plotHarvest(self, show_max_harvest=False, show_original=True, daily=False, n_random=0):
        fig, ax = plt.subplots()
        ax.set(xlabel="harvest day" if daily else "harvest week", ylabel="harvest quantity")
        if n_random > 0:
            H = self.getRandomHarvestMatrices(n_random)
        if show_original:
            if n_random > 0:
                wh_orig = self.getHarvest(self.orig_days, daily)[0]
                wh_orig_rand = self.getRandomHarvest(H, self.orig_days, daily)
                x_orig  = np.arange(len(wh_orig))
                ax.bar(x_orig, wh_orig, alpha=0.3, label="orig. (mean GDU)", color="C1")
                ax.errorbar(x_orig, np.mean(wh_orig_rand, axis=0), np.std(wh_orig_rand, axis=0, ddof=1), alpha=0.4, label="orig. (rand. GDU)", fmt="_", color="C1")
            else:
                wh_orig = self.getHarvest(self.orig_days, daily)[0]
                x_orig  = np.arange(len(wh_orig))
                ax.bar(x_orig, wh_orig, alpha=0.2, label="orig. (rand. GDU)", color="C1")

        if show_max_harvest:
            wh_max  = self._getMaxWeeklyHarvest()
            x_max   = np.arange(len(wh_max))
            ax.scatter(x_max, wh_max, marker="_", label=f"$h_{{max}}$")
        
        if np.any(self.loss):
            wh_opti = self.getHarvest(self.best_planting_days, daily)[0]
            if n_random > 0:
                wh_opti_rand = self.getRandomHarvest(H, self.best_planting_days, daily)
                wh_opti_mean = np.mean(wh_opti_rand, axis=0)
                wh_opti_std = np.std(wh_opti_rand, axis=0, ddof=1)
                
            x_opti = np.arange(len(wh_opti))
            if self.capacity_limit:
                cap = self.capacity_limit/7. if daily else self.capacity_limit
                max_perc_hi  = 100*(np.max(wh_opti/cap)-1)
                max_capacity = np.max(wh_opti)
                abs_max_diff = np.max(np.abs(wh_opti[wh_opti>0]-cap))
                abs_med_diff = np.median(np.abs(wh_opti[wh_opti>0]-cap))
                n_weeks      = np.sum(wh_opti > 0)
                n_over_cap   = np.sum(wh_opti > cap)
                info_str =(f"$OR=$ {max_perc_hi:.2f} %\n"
                           f"$\Delta_{{max}}=$ {abs_max_diff:.2f}\n"
                           f"$\Delta_{{median}}=$ {abs_med_diff:.2f}\n"
                           f"$n_h=$ {n_weeks:0g}\n"
                           f"$n_{{>C}}=$ {n_over_cap:0g}")
            else:
                if self.scenario == 2:
                    target_cap = np.sum(wh_opti)/np.sum(wh_opti>0)
                elif self.scenario == 3:
                    target_cap = np.sum(wh_opti)/57.
                
                max_perc_hi  = 100*(np.max(wh_opti/target_cap)-1)
                max_capacity = np.max(wh_opti)
                abs_max_diff = np.max(np.abs(wh_opti[wh_opti>0]-target_cap))
                abs_med_diff = np.median(np.abs(wh_opti[wh_opti>0]-target_cap))
                n_weeks      = np.sum(wh_opti > 0)
                n_over_cap   = np.sum(wh_opti > target_cap)
                info_str =(f"$OR=$ {max_perc_hi:.2f} %\n"
                           f"$\Delta_{{max}}=$ {abs_max_diff:.2f}\n"
                           f"$\Delta_{{median}}=$ {abs_med_diff:.2f}\n"
                           f"$n_h=$ {n_weeks:0g}\n"
                           f"$n_{{>C}}=$ {n_over_cap:0g}")

            ax.bar(x_opti, wh_opti, alpha=0.6, label="opt. (mean GDU)\n"+info_str, color="C0")
            if n_random > 0:
                ax.errorbar(x_opti, wh_opti_mean, wh_opti_std, alpha=1, label="opt. (random GDU)", fmt="_", color="C0")
        
        ax.axhline(max_capacity, c="C4", ls="--", label=f"$C_{{need}}=$ {max_capacity:.2f}")
        if self.capacity_limit:
            ax.axhline(self.capacity_limit/7. if daily else self.capacity_limit, c="C3", ls="--", label=f"$C_{{target}}=${self.capacity_limit:.2f}")
        else:
            if self.scenario == 2:
                ax.axhline(target_cap, c="C3", ls="--", label=f"$C_{{2-1}}=$ {target_cap:.2f}")
            elif self.scenario == 3:
                ax.axhline(target_cap, c="C3", ls="--", label=f"$C_{{2-2}}=$ {target_cap:.2f}")
            ax.axhline()
        ax.legend()
        ax.grid()
        ax.set(ylim=(0,None))
        
        return fig, ax


def calc_hl(h, capacity):
    h_scaled = np.asarray(h)/capacity
    loss = ( np.sum(np.where(h_scaled >1, np.exp(h_scaled)-np.e, 0)),
                        np.sum(np.where(h_scaled<=1, h_scaled*(1-h_scaled), 0)))          
    return loss


class HarvestOptimizationMOO(Problem):

    def __init__(self, site, scenario, cyclic_years):
        self.o = HarvestOptimization(data_dir="data/")
        self.o.loadSiteScenario(site, scenario, cyclic_years)
        super().__init__(   n_var = len(self.o.orig_days),
                            n_constr = 0,
                            n_obj = 6,   
                            xl = self.o.planting_bounds[:,0].astype(int),
                            xu = self.o.planting_bounds[:,1].astype(int))
        

    def _evaluate(self, X, out, *args, **kwargs):        
        h = np.stack([self.o.getHarvest(x)[0] for x in X])
        C = self.o.capacity_limit
        OR  = 100*(np.max(h/C-1, axis=1))
        Cmax = np.max(h, axis=1)
        Delta_max = np.stack([np.max(np.abs(h_[h_>0]-C)) for h_ in h])
        Delta_med = np.stack([np.median(np.abs(h_[h_>0]-C)) for h_ in h])
        nh = np.sum(h>0, axis=1)
        no = np.sum(h>C, axis=1)

        out["F"] = np.column_stack([OR, Cmax, Delta_max, Delta_med, nh, no])


class HarvestOptimizationHLO(Problem):

    def __init__(self, site, scenario, cyclic_years):
        self.o = HarvestOptimization(data_dir="data/")
        self.o.loadSiteScenario(site, scenario, cyclic_years)
        self.capacity = self.o.capacity_limit
        super().__init__(   n_var = len(self.o.orig_days),
                            n_constr = 0,
                            n_obj = 6,   
                            xl = self.o.planting_bounds[:,0].astype(int),
                            xu = self.o.planting_bounds[:,1].astype(int))
        

    def _evaluate(self, X, out, *args, **kwargs):        
        h = np.stack([self.o.getHarvest(x)[0] for x in X])
        L = [calc_hl(h_, self.capacity) for h_ in h]

        out["F"] = np.stack(L)


class MySampling(Sampling):

    def __init__(self, site, scenario, cyclic_years):
        super().__init__()
        self.o = HarvestOptimization(data_dir="data/")
        self.o.loadSiteScenario(site, scenario, cyclic_years)

    def _do(self, problem, n_samples, **kwargs):
        X = np.stack([self.o.getRandomPlantingDays() for _ in range(n_samples)])
        return X


class MyMutation(Mutation):

    def __init__(self, site, scenario, cyclic_years):
        super().__init__()
        self.o = HarvestOptimization(data_dir="data/")
        self.o.loadSiteScenario(site, scenario, cyclic_years)
        self.mutation_prob = 1/500
        
    def _do(self, problem, X, **kwargs):
        np.random.seed()
        while True:
            change = np.random.random(len(self.o.planting_ranges)) < self.mutation_prob
            if np.sum(change) > 0:
                break

        X_mut = X.copy()
        X_mut[0,change] = np.array([np.random.randint(low, up+1) for low, up in self.o.planting_bounds[change]], dtype=int)
        return X_mut
