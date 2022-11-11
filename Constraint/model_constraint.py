import numpy as np
import pandas as pd
import kilonovanet
import astropy.units as u
import astropy.constants as c
from astropy.time import Time

MPC_CGS = u.Mpc.cgs.scale
t = '2020-01-05T16:24:26'
mjd = Time(t,format='isot').mjd

class model_constraint(object):
    #By now surrogate model kilonovanet are used to constraint
    def __init__(self,event_dir,mode,n,id_list):
        self.mode = mode
        self.n = n
        self.id_list = id_list
        #ZTF Filter
        self.filter_map = {
            '1':'ZTF_g',
            '2':'ZTF_r',
            '3':'ZTF_i'
        }

        self.event_dir = str(event_dir)
        #trigger time
        if self.event_dir.split('/')[-1] == 'ZTF_fields_LALInf_S200105ae.dat':
            self.ref_time = 58853.683634
        elif self.event_dir.split('/')[-1] == 'ZTF_fields_LALInf_S200115j.dat':
            self.ref_time = 58863.182754
        ###EVENT NAME
        self.event_name = self.event_dir.split('/')[-1].split('.')[0].split('_')[-1]
        raw_data = self.load_event()
        self.data = raw_data[raw_data['Filter'].isin(self.id_list)]

        self.ZTF_field = pd.read_csv('/home/Aujust/data/Kilonova/possis/Model Constraints/ID.csv', index_col=0)

        self.model = kilonovanet.Model('/home/Aujust/data/Kilonova/possis/data/metadata_bulla_bhns.json',
                 '/home/Aujust/data/Kilonova/possis/models/bulla-bhns-latent-2-hidden-500-CV-4-2021-04-17-epoch-200.pt',
                 filter_library_path='/home/Aujust/data/Kilonova/possis/data/filter_data')

        self.extinction()
        self.bin(mode,n)
        print('Event data loaded!')
   
    def load_event(self):
        #Filter ID: g-1 r-2 i-3
        f = pd.read_csv(self.event_dir, header=None, sep=' ', names=['ID','Filter','Time','Mag'])
        times = list(f['Time'])
        t = Time(times, format='isot')
        f['Time'] = t.mjd - self.ref_time
        return f

    def extinction(self):
        #refer the mapping file https://github.com/skyportal/skyportal/blob/main/data/ZTF_Fields.csv
        ebv_R = {
            'u':3.995,
            'g':3.214,
            'r':2.165,
            'i':1.592,
            'z':1.211
        }

        ebv_R = [[1,3.2],[2,2.2],[3,1]]
        ebv_R = pd.DataFrame(ebv_R, columns=['Filter','Rv'])
        assemble_data = self.data.merge(self.ZTF_field[['ID','Ebv']])
        assemble_data = assemble_data.merge(ebv_R, how='left')
        assemble_data['Mag'] -= assemble_data['Ebv']*assemble_data['Rv']
        self.data = assemble_data


    def bin(self,mode,n):
        #Implemented using K-Means
        #mode: median or best
        #n: number of binned data
        def median(df):
            df2 = df.sort_values(by='Mag',ascending=False)
            return df2.iloc[int(len(df2.index)/2),:]
        def deepest(df):
            df2 = df.sort_values(by='Mag',ascending=False)
            return df2.iloc[0,:]
        from sklearn.cluster import KMeans
        X = self.data['Time'].to_numpy().reshape(-1,1)
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        labels = kmeans.labels_
        self.data['Label'] = labels
        if mode == 'median':
            self.bin_data = self.data.groupby(['Filter','Label']).apply(median)
        elif mode == 'deepest':
            self.bin_data = self.data.groupby(['Filter','Label']).apply(deepest)
        else:
            print('No mode matched!')

        #self.kmeans_data = self.data.groupby(['Filter','Label']).agg(mode_arg[self.mode])
        #idx = self.data[self.data['Filter']==self.filter_id].groupby(['Filter','Label'])['Mag'].idxmax()

        #self.bin_data = self.data.iloc[idx]

    def get_redshift(self,dL):
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=67.80, Om0=0.308)
        redshift = 0
        err = 1
        red_list = np.linspace(0.001,0.1,500)
        for red in red_list:
            err0 = abs(cosmo.luminosity_distance(red).value-dL)
            if err0 < err:
                err = err0
                redshift = red
        redshift = round(redshift,6)
        return redshift

    def add_params(self,params):
        #Dict(key:List())
        self.free_params = params

    def constraint(self):
        kwargs = ['M_{dyn}','M_{pm}','cos(\Theta_{obs})']
        self.param_name = kwargs
        param_list = []
        param_allow = dict()
        param_deny = dict()
        for name in kwargs:
            if name in kwargs:
                param_list.append(self.free_params[name])

        #loop for all param space
        #This is a brief version just loop mass and fix Phi and Theta
        times = self.bin_data['Time'].values.flatten()
        obs_mags = self.bin_data['Mag'].values.flatten()
        self.times = times
        self.obs_mags = obs_mags

        bands = self.data.loc[self.data['Time'].isin(times),['Filter']].values.flatten()     
        bands = np.array([self.filter_map[str(i)] for i in bands])

        dL_list = np.array([-1,0,1])*self.free_params['ddL'] + self.free_params['dL']
        self.dL_list = dL_list
        ### COLOR CONTROL
        color_list = ['lightskyblue','deepskyblue','dodgerblue']
        color_map = dict()
        for i,dL in enumerate(dL_list):
            param_allow[str(dL)] = []
            param_deny[str(dL)] = []
            color_map[str(dL)] = color_list[i]
            z = self.get_redshift(dL)
            times = times * (1.0 + z)

            for M_dyn in param_list[0][:]:
                for M_pm in param_list[1][:]:
                    for cos_theta in param_list[2][:]:
                        param = np.array([M_dyn,M_pm,cos_theta])
                        mags = self.model.predict_magnitudes(param,times=times,filters=bands,distance=dL*MPC_CGS)

                        if np.min(mags-obs_mags) <= -0.1:
                                param_deny[str(dL)].append(param)
                        else:
                                param_allow[str(dL)].append(param)
        self.allow = param_allow
        self.deny = param_deny
        self.color_map = color_map
        print('Constraint Completed!')

    def plot_lc(self,filter_ID):
        print('Drawing Light Curves!')
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines


        matplotlib.rcParams['mathtext.rm'] = 'serif'
        matplotlib.rc('font', family='serif', size=14)
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 300

        fig = plt.figure(figsize=(9,5))
        ax = fig.gca()
        labels = []
        labelcolors = []
        #plot data points
        f = self.data
        ax.plot(f[f['Filter']==filter_ID]['Time'],f[f['Filter']==filter_ID]['Mag'],'v',markerfacecolor='none',
            label=self.filter_map[str(filter_ID)],alpha=0.8,color='lawngreen',zorder=3)
        labels += ['Observation']
        labelcolors += ['lawngreen']
        #ax.plot(f1.loc[(filter_ID,np.arange(0,self.n,1)),['Time']].values.flatten(),f1.loc[(filter_ID,np.arange(0,self.n,1)),['Mag']].values.flatten(),
        #    'v',color='darkolivegreen',zorder=4)
        ax.plot(self.bin_data.loc[(filter_ID,np.arange(0,self.n,1)),'Time'],self.bin_data.loc[(filter_ID,np.arange(0,self.n,1)),'Mag'],'v',color='darkolivegreen',zorder=4)

        labels += [self.mode]
        labelcolors += ['darkolivegreen']

        times = np.linspace(0,5,100)
        bands = np.repeat([self.filter_map[str(filter_ID)]],len(times))
        for dL,deny in self.deny.items():
            for param in deny:
                mags = self.model.predict_magnitudes(param,times=times,filters=bands,distance=float(dL)*MPC_CGS)
                ax.plot(times,mags,alpha=0.3,color = self.color_map[dL],label=str(dL),zorder=2)

        for dL,allow in self.allow.items():
            for param in allow:
                mags = self.model.predict_magnitudes(param,times=times,filters=bands,distance=float(dL)*MPC_CGS)
                ax.plot(times,mags,alpha=0.1,color = 'lightsteelblue',zorder=1)
        labels += self.color_map.keys()
        labelcolors += self.color_map.values()
        for i in range(2,len(labels)):
            labels[i] = 'Ruled out at ' + labels[i] + ' Mpc'

        ax.set_xlim(1e-1,times[-1])
        ax.set_ylim(23,16)
        ax.set_xscale('log')
        ax.set_xlabel('Time Since Merger')
        ax.set_ylabel(self.filter_map[str(filter_ID)][-1]+' (AB Mag)')
        handles = []
        for i in range(len(labels)):
            handle = mlines.Line2D([], [], color=labelcolors[i], marker='s', ls='', label=labels[i])
            handles.append(handle)
        plt.legend(handles=handles,fontsize=10)
        plt.title(self.event_name)
        plt.savefig('possis/Model Constraints/lc_'+self.event_name+'_'+self.filter_map[str(filter_ID)][-1]+'_'+self.mode+'_'+str(self.free_params['cos(\Theta_{obs})'][0])+'.jpg')
        print('Light Curve Drawed!')

    def plot_params(self):
        print('Drawing prameter plane!')
        #Default plot M_dyn - M_pm plane
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import copy

        matplotlib.rcParams['mathtext.rm'] = 'serif'
        matplotlib.rc('font', family='serif', size=14)
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 300

        XX = self.free_params[self.param_name[0]]
        YY = self.free_params[self.param_name[1]]

        data = copy.copy(self.deny)
        for key,value in data.items():
            data[key] = pd.DataFrame(value,columns=self.param_name)

        contour = [np.zeros((len(self.free_params[self.param_name[0]]),len(self.free_params[self.param_name[0]])))] * 3
        colors = ['white']
        levels = [0]

        fig,ax = plt.subplots(1,3,figsize=(16,5))
        min_list = [0,0.6,0.9]
        for i in range(3):
            #theta_min = i*0.4
            #theta_max = theta_min + 0.2
            theta_min = min_list[i]
            theta_max = theta_min + 0.1
            for dL in self.dL_list:
                if i == 0:
                    levels.append(dL-1)
                    colors.append(self.color_map[str(dL)])
                deny_dL = data[str(dL)]
                deny_dL2 = deny_dL.loc[(deny_dL['cos(\Theta_{obs})']>theta_min) & (deny_dL['cos(\Theta_{obs})']<theta_max),:]

                dL = float(dL)
                if len(deny_dL2.index) > 0:
                    for j in range(len(deny_dL2.index)):
                        ix = np.where(XX==deny_dL2.iloc[j,0])[0][0]
                        iy = np.where(YY==deny_dL2.iloc[j,1])[0][0]
                        if dL > contour[i][ix,iy]:
                            contour[i][ix,iy] = dL
            if i == 0:
                levels.append(dL+100)

            ax[i].contourf(XX,YY,contour[i].T,levels=levels,colors=colors)
            ax[i].set_xlabel('$'+self.param_name[0]+'(M_{\odot})'+'$')
            if i == 0:
                ax[i].set_ylabel('$'+self.param_name[1]+'(M_{\odot})'+'$')
            ax[i].set_title('$'+format(theta_min,'.1f')+'<'+self.param_name[-1]+'<'+format(theta_max,'.1f')+'$')

        plt.suptitle(self.event_name)
        plt.savefig('possis/Model Constraints/param_'+self.event_name+'_'+self.mode+'.jpg')
        print('Parameter plane drawed!')


params = {
    'M_{dyn}':np.linspace(1e-2,9e-2,10),
    'M_{pm}':np.linspace(1e-2,9e-2,10),
    '\Phi':[30],
    'cos(\Theta_{obs})':np.linspace(0,1,20),
    'dL':340,
    'ddL':79
}


file_name = '/home/Aujust/data/Kilonova/possis/Model Constraints/ZTF_fields_LALInf_S200105ae.dat'
A = model_constraint(file_name,'deepest',5,id_list=[1,2])
A.add_params(params)
A.constraint()
A.plot_params()
A.plot_lc(1)
A.plot_lc(2)

print()
