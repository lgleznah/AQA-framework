import itertools

def main():
    header = "obj,mod,pre,tsize,vsize,net,act,loss,opt,lr,bsize,epochs,dropout,clean_tech,retrain_mode"

    objs_and_mods = ["distribution,none", "mean,binaryWeights"]
    nets = ["mobilenet", "inception", "vgg16"]
    optimizers = ["Adam", "SGD"]
    lrs = [1e-4, 1e-5, 1e-6]

    grid = [objs_and_mods, nets]

    with open("experiments.csv", "w") as f:
        print(header, file=f)

        for conf in itertools.product(*grid):            
            print(f"{conf[0]},none,0.08,0.2,{conf[1]},softmax,{'emd' if conf[0][0] == 'd' else 'cross'},Adam,1e-3,128,20,0.75,none,none", file=f)
            print(f"{conf[0]},none,0.08,0.2,{conf[1]},softmax,{'emd' if conf[0][0] == 'd' else 'cross'},Adam,1e-7,128,20,0.75,none,none", file=f)
            print(f"{conf[0]},none,0.08,0.2,{conf[1]},softmax,{'emd' if conf[0][0] == 'd' else 'cross'},Adam,1e-5,128,20,0.6,none,none", file=f)
            print(f"{conf[0]},none,0.08,0.2,{conf[1]},softmax,{'emd' if conf[0][0] == 'd' else 'cross'},Adam,1e-5,128,20,0.9,none,none", file=f)

if __name__ == "__main__":
    main()