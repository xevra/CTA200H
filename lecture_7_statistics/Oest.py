import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
rng = np.random.RandomState()
KRAKEN_HP = 84
KRAKEN_DR = 5
TRIALS = 20
#TRIALS = 200
#TRIALS = 2000
#TRIALS = 20000
#TRIALS = 20000000

def roll_1d4r1e4(keep=False,num=1):
    value = rng.randint(low=1,high=5,size=num)
    if np.any(value == 1) and not keep:
        value[value == 1] = roll_1d4r1e4(
            keep=True,
            num=np.count_nonzero(value == 1),
        )
    if np.any(value == 4):
        value[value == 4] += roll_1d4r1e4(
            keep=False,
            num=np.count_nonzero(value == 4),
        )
    return value

def crossbow_bolt(num=1):
    first   = roll_1d4r1e4(num=num)
    second  = roll_1d4r1e4(num=num)
    third   = roll_1d4r1e4(num=num)
    dammage = first + second + third
    dammage = dammage - KRAKEN_DR
    dammage[dammage < 0] = 0
    return dammage

def slay_kraken(num=1):
    current_hp = np.full(num,KRAKEN_HP)
    counts = np.zeros(num,dtype=int)
    while np.any(current_hp > 0):
        mask = current_hp > 0
        counts[mask] += 1
        current_hp[mask] -= crossbow_bolt(num=np.count_nonzero(current_hp > 0))
    return counts

def plot_statistics(random_sample):
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    ax.hist(
        random_sample,
        bins=np.max(random_sample)+1 - np.min(random_sample),
        range=[np.min(random_sample),np.max(random_sample)],
        density=True,
    )
    sample_mean = np.sum(random_sample) / TRIALS
    sample_var = np.sum((random_sample - sample_mean)**2)/(TRIALS - 1)
    sample_std = np.sqrt(sample_var)
    x = np.linspace(
        start=np.min(random_sample),
        stop  = np.max(random_sample),
        num =10*(np.max(random_sample)-np.min(random_sample))
    )
    ax.plot(x, norm.pdf(x, loc=sample_mean, scale=sample_std))
    plt.show()

def main():
    print(f"Kraken HP: {KRAKEN_HP}")
    print(f"Kraken DR: {KRAKEN_DR}")
    print(f"Trials: {TRIALS}")
    random_sample = slay_kraken(num=TRIALS)
    sample_mean = np.sum(random_sample) / TRIALS
    print(f"Minimum number of hits: {np.min(random_sample)}")
    print(f"Maximum number of hits: {np.max(random_sample)}")
    print(f"Sample mean: {sample_mean}")
    sample_var = np.sum((random_sample - sample_mean)**2)/(TRIALS - 1)
    print(f"Sample Var: {sample_var}")
    standard_deviation = np.sqrt(sample_var)
    print(f"Std: {standard_deviation}")
    plot_statistics(random_sample)

if __name__ == "__main__":
    main()
