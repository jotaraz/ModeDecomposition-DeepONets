import os
import numpy as np 
import sys 

SVD_cfg = {
    'Nepochs'    : 4000,
    'vtag'       : 0,
    'depth'      : 5,
    'width'      : 335,
    'llw'        : 50,
    'doplot'     : 0,
    'batch_name' : 'kdvnx401_dt0.0001_nc5_m5000',
    'lrstag'     : '32',
    'init_lr'    : 1e-4,
    'decay_rate' : 0.95,
    'num_data'   : 1000,
    'which_T'    : 0,
    'dotruesigma': 1,
    'uendtag'    : '1999',
    'sigmascale' : 1.0,
    'exponent'   : 1.0,
    'doadam'     : 1,
    'dostacked'  : 0,
}

NormalDON_cfg = {
    'Nepochs'    : 4000,
    'vtag'       : 0,
    'depth'      : 5,
    'width'      : 335,
    'llw'        : 50,
    'doplot'     : 0,
    'batch_name' : 'kdvnx401_dt0.0001_nc5_m5000',
    'lrstag'     : '32',
    'init_lr'    : 1e-4,
    'decay_rate' : 0.95,
    'num_data'   : 1000,
    'which_T'    : -1,
    'dotruesigma': 0,
    'uendtag'    : '1999',
    'sigmascale' : 'First',
    'exponent'   : 1.0,
    'doadam'     : 1,
    'dostacked'  : 0,
}


def generate_cfg(N, d, w, batch_name='kdvnx401_dt0.0001_nc5_m5000', uendtag='1999', do_don=True):
    if do_don:
        which_T = -1
        dotruesigma = 0
        sigmascale = 'First'
    else:
        which_T = 0
        dotruesigma = 1
        sigmascale = 1.0
    cfg = {
        'Nepochs'    : 4000,
        'vtag'       : 0,
        'depth'      : d,
        'width'      : w,
        'llw'        : N,
        'doplot'     : 0,
        'batch_name' : batch_name,
        'lrstag'     : '32',
        'init_lr'    : 1e-4,
        'decay_rate' : 0.95,
        'num_data'   : 1000,
        'which_T'    : which_T,
        'dotruesigma': dotruesigma,
        'uendtag'    : uendtag,
        'sigmascale' : sigmascale,
        'exponent'   : 1.0,
        'doadam'     : 1,
        'dostacked'  : 0,
    }
    return cfg

def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 50
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 50
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 50
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 50    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 50
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 50
    if tag == 8:
        return "NormalHO50_.2to.9_exTo1.1_m1000", "1", 20
    if tag == 9:
        return "waveeq1_N10_m1000", "-3.0", 10 
    if tag == 10:
        return "waveeq1_N10_m1000", "-1.0", 10 
    if tag == 11:
        return "waveeq1_N10_m1000", "-0.1", 10 
    if tag == 12:
        return "waveeq1_N10_m1000", "0.1", 10 
    if tag == 13:
        return "waveeq1_N10_m1000", "1.0", 10 
    if tag == 14:
        return "waveeq1_N10_m1000", "3.0", 10 
    if tag == 15:
        return "waveeq1f_N10_m1000", "-3.0", 10 
    if tag == 16:
        return "waveeq1f_N10_m1000", "-1.0", 10 
    if tag == 17:
        return "waveeq1f_N10_m1000", "-0.1", 10 
    if tag == 18:
        return "waveeq1f_N10_m1000", "0.1", 10 
    if tag == 19:
        return "waveeq1f_N10_m1000", "1.0", 10 
    if tag == 20:
        return "waveeq1f_N10_m1000", "3.0", 10 

def lrs(lrtag):
    if lrtag == 32:
        return 1e-4, 0.95
    elif lrtag == 34:
        return 5e-4, 0.95
    elif lrtag == 38:
        return 1e-3, 0.95
    elif lrtag == 40:
        return 2e-3, 0.95
    elif lrtag == 42:
        return 4e-3, 0.95
    elif lrtag == 43:
        return 8e-3, 0.95

# for vtag in range(0,1):
#     #for lrstag, lr, decay in zip(lrstags, lrs, decays):
#     for lrstag in lrstags:
#         lr, decay = lrs(lrstag)
#         for iw in range(len(gen_widths)):
#             for depth in depths:
#                 for tag in btags:
#                     for (which_T, dotruesigma, sigmascale) in zip(whTs, dtss, sisc):
#                         if tag < 2:
#                             llw = 20
#                         else:
#                             llw = 50
#                         batch_name, endtag, _ = dic(tag)

#                         doadam    = 0
#                         dostacked = gen_stacked[iw]
#                         width = gen_widths[iw]

def run(cfg):
    dwllw = str(cfg['depth'])+" "+str(cfg['width'])+" "+str(int(cfg['llw']))
    tmp  = str(cfg['Nepochs'])+" "+str(cfg['vtag'])+" "+dwllw+" 0 "+cfg['batch_name']+" "
    tmp += str(cfg['lrstag'])+" "+str(cfg['init_lr'])+" "+str(cfg['decay_rate'])+" "+str(cfg['num_data'])+" "
    tmp += str(cfg['which_T'])+" "+str(cfg['dotruesigma'])+" "+str(cfg['uendtag'])+" "+str(cfg['sigmascale'])+" "
    tmp += str(cfg['exponent'])+" "+str(cfg['doadam'])+" "+str(cfg['dostacked'])
    os.system("python execute_don.py "+tmp)


# Model grid: for each (llw=N, depth=d) size and width w, train both a
# learned-trunk DeepONet (do_don=True) and an SVD-basis model (do_don=False).
Ns = [50, 100, 100]
ds = [10, 5, 10]

CONFIGS = []
for (N, d) in zip(Ns, ds):
    for w in [100, 500]:
        for do_don in (True, False):
            CONFIGS.append(generate_cfg(N, d, w, do_don=do_don))

if __name__ == "__main__":
    # `python run_grid.py --count`  -> number of configs (for the condor queue)
    # `python run_grid.py <index>`  -> run exactly ONE config (HTCondor fan-out)
    # `python run_grid.py`          -> run all configs sequentially (local)
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(CONFIGS))
    elif len(sys.argv) > 1:
        run(CONFIGS[int(sys.argv[1])])
    else:
        for cfg in CONFIGS:
            run(cfg)
