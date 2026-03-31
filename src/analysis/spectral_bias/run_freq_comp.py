import os 

for i in [0, 1, 3, 4, 5, 6, 7]:
    if i < 2:
        a = 0
        #os.system("python freq_comparison.py "+str(i)+" 20")
    if i > 2:
        #os.system("python freq_comparison.py "+str(i)+" 50")
        os.system("python freq_comparison.py "+str(i)+" 100")