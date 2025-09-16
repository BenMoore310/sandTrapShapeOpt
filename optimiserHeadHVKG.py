import numpy as np
from scipy.stats import qmc
import argparse
import matplotlib.pyplot as plt
import json
import subprocess
import sandTrapCatmull as STC
import HVKGHydroshield
import math
# plt.style.use(["science", "notebook"])

# Some functions for handling the fact that x coords get ordered before simulation:
# for each simulation based on n pairs of x,y coordinates, the one function value returned
# gives the result for n! feasible designs. 

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

# Function to find the possible permutations. 
# Initial value of idx is 0.
def permutations(res, arr, idx):
  
    # Base case: if idx reaches the size of the array,
    # add the permutation to the result
    if idx == len(arr):
        res.append(arr[:])
        return

    # Permutations made by swapping each element
    for i in range(idx, len(arr)):
        swap(arr, idx, i)
        permutations(res, arr, idx + 1)
        swap(arr, idx, i)  # Backtracking

# Function to get the permutations
def permute(arr):
    res = []
    permutations(res, arr, 0)
    return res



cwdPath = "/home/bm424/Projects/sandTrapShapeOpt"


def simulateDesign(objective, sample, numBasis):

    """
    Simulate the design using the current parameters.
    """

    print(objective, sample, numBasis)

    STC.main(sample, numBasis)




    objective_strings = ["075", "125"]
    directoryString = objective_strings[objective]

    src = f"0_timesteps/0.{directoryString}"
    dst = "./0"

    cloudSrc = f"cloudPropertiesFiles/kinematicCloudProperties{directoryString}"
    cloudDst = "constant/kinematicCloudProperties"

    subprocess.run(["cp", "-r", src, dst], cwd=cwdPath + "/sandTrapCaseDir", check=True)

    # subprocess.run(
    #     ["cp", cloudSrc, cloudDst], cwd=cwdPath + "/runDirectory", check=True
    # )

    # Placeholder for simulation logic
    print("Simulating design with current parameters...")

    subprocess.run(["bash", "allrun"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

    print("Checking simulation stability...")

    subprocess.run(
        ["python3.11", "checkError.py"], cwd=cwdPath + "/sandTrapCaseDir", check=True
    )    

    maxCo = np.loadtxt(cwdPath + "/sandTrapCaseDir/maxCo.txt")

    # assume simulation is stable if maxCo less than 1000, and proceed with the rest of the particle tracking
    if maxCo < 1e3:

        print('Solution stable. Proceeding...')

        result = subprocess.run(["bash", "allrun2"], cwd=cwdPath + "/sandTrapCaseDir", check=False)

        if result.returncode == 42:
            numberEfficiency = 20

            # subprocess.run(["rm", "-r", "0"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

            # subprocess.run(["rm", "-r", "processor*"], cwd=cwdPath + "/runDirectory", check=True)
            subprocess.run(["bash", "allclean"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

            return numberEfficiency

        elif result.returncode == 0:
            
            print('Simulation successful, calculating true efficiency...')

            subprocess.run(
                ["python3.11", "calculateDesignEfficiency.py"], cwd=cwdPath + "/sandTrapCaseDir", check=True
            )

            efficiencies = np.loadtxt(cwdPath + "/sandTrapCaseDir/efficiency.txt")

            numberEfficiency = efficiencies[0]
            massEfficiency = efficiencies[1]
            print('efficiency = ', numberEfficiency)

            # subprocess.run(["rm", "-r", "0"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

            # subprocess.run(["rm", "-r", "processor*"], cwd=cwdPath + "/runDirectory", check=True)
            subprocess.run(["bash", "allclean"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

            # numberEfficiency = np.random.randint(0,10)

            return numberEfficiency
    
    else:
        # if maxCo greater than 1000, assume instability and penalise decision vector by returning an arbitrarily low number efficiency
        print('Solution unstable, penalising decision vector...')
        numberEfficiency = 20.0

        # subprocess.run(["rm", "-r", "0"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

        # subprocess.run(["rm", "-r", "processor*"], cwd=cwdPath + "/runDirectory", check=True)
        subprocess.run(["bash", "allclean"], cwd=cwdPath + "/sandTrapCaseDir", check=True)

        return numberEfficiency




def main(numBasis, numObj, initialSamples, seed):



    # list of arrays of values for each objective
    featureList = [np.array([]) for _ in range(numObj)]
    targetList = [np.array([]) for _ in range(numObj)]



    # evaluatedObjectives = np.random.randint(
    #     low=0, high=6, size=(initialSamples), dtype=int
    # )

    lbX = 3.7
    lbY = -1.54

    ubX = 23.5
    ubY = 1


    bounds = []

    for i in range(numBasis):
        bounds.append([lbX, ubX])
        bounds.append([lbY, ubY])


    bounds = np.array(bounds)

    lowBounds = bounds[:, 0]
    highBounds = bounds[:, 1]

    # Generate Latin Hypercube samples
    # TODO add a check later on where if the number of bases is 1, then the weight parameter is set to 1.0
    # if numBasis == 1:
    #     sampler = qmc.LatinHypercube(d=(numBasis * 2), seed=seed)
    # else:
    #     sampler = qmc.LatinHypercube(d=(numBasis * 3), seed=seed)

    # uncomment from here for the initial solutions:

    # for i in range(numObj):
    #     sampler = qmc.LatinHypercube(d=len(bounds), seed=seed+i)

    #     # remember this is initial samples per objective, not in total
    #     # init per obj should be 2*D + 1

    #     samples = sampler.random(n=initialSamples)

    #     # Scale samples to bounds
    #     featureList[i] = qmc.scale(samples, lowBounds, highBounds)

    # print("initialPopulation", featureList)


    # for objIdx in range(numObj):
    #     initialPopulation = featureList[objIdx]
    #     for sample in initialPopulation:

    #         i = 0


    #         print("sample", sample)
    #         print(sample.shape)
    #         # print(np.reshape(sample, (numBasis,2)))

    #         # call STC and generate spline from current sample

    #         numberEfficiency = simulateDesign(objIdx, sample, numBasis)

    #         # targetList[objIdx] = np.append(targetList[objIdx], numberEfficiency)

    #         targetList[objIdx] = np.append(targetList[objIdx], numberEfficiency)

    #         print('Target list: ', targetList)

    #         # sample_alt = np.reshape(sample, (int(len(sample)/2), 2))
    #         # res = np.array(permute(list(sample_alt[:,0])))
    #         # print(len(res))
    #         # print(res.shape)

    #         # sample_final = []

    #         # for j in range(0, len(res)):
    #         #     linkedArray = np.reshape(np.vstack((res[j], sample_alt[:,1])).T, (-1,))
    #         #     sample_final.append(linkedArray)

    #         # sample_final = np.array(sample_final)
            
    #         # targets_alt = np.full((len(res),), fill_value=numberEfficiency)

    #         # featureListFull[objIdx] = np.vstack((featureListFull[objIdx], sample_final))
    #         # print(featureListFull[objIdx])
    #         # targetListFull[objIdx] = np.append(targetListFull[objIdx], targets_alt)
    #         # print(targetListFull[objIdx])

    #         i += 1

    # # # print(np.array(featureListFull).shape)
    # # # featureListFull = np.array(featureListFull)
    # # for objIdx in range(numObj):
    # #     print(objIdx)
    # #     featureListFull[objIdx] = np.delete(featureListFull[objIdx], 0,0)

    # # print(featureListFull, np.array(featureListFull).shape)

    # # print(targetList)
    # # # print(featureList)
    # # # print(featureListFull)
    # # # print(np.array(featureListFull[0]).shape)
    # # # print(targetListFull)

    # np.savetxt('targetList.txt', targetList)
    # arr = np.vstack(featureList)   # shape (numObj, length_of_each_array)
    # np.savetxt("features.txt", arr)    

    featureList = np.reshape(np.loadtxt('features.txt'), (numObj,(initialSamples),numBasis*2))
    targetList = np.loadtxt('targetList.txt')
    HVKG = HVKGHydroshield.HVKG()
    HVKG.optimise(bounds.T, simulateDesign, featureList, targetList.T)

# python3.11 optimiserHeadHVKG.py --numBasis 4 --numObj 2 --initialSamples 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the head script for HydroShield optimisation."
    )
    parser.add_argument(
        "--numBasis",
        type=int,
        default=2,
        help="Number of Beta basis functions to use in the curve generation.",
    )
    parser.add_argument(
        "--numObj",
        type=int,
        default=1,
        help="Number of objectives to be optimised over.",
    )
    parser.add_argument(
        "--initialSamples",
        type=int,
        default=5,
        help="Number of initial samples for the Latin Hypercube sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Sets the seed for the initial LHS generation"
    )


    args = parser.parse_args()

    main(args.numBasis, args.numObj, args.initialSamples, args.seed)
