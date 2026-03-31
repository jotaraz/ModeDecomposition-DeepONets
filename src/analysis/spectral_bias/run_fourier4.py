import os

for bid in [3]:
    for kmax in [1.0, 10.0]: #[0.1, 0.3, 1.0, 3.0, 10.0]:
        for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]: #, 18, 22, 26, 30]: #[0, 1, 2, 3, 4, 5, 6]:
            print("bid", bid, "i", i, "kmax", kmax)
            os.system("python fourier_v4.py "+str(bid)+" "+str(i)+" "+str(kmax)+" 5_000")



