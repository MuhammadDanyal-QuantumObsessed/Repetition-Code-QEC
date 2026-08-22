"""
Repetition Code — Quantum Error Correction Simulation

Converted from "Another copy of Repetition Code.ipynb"
Uses Stim (circuit simulation), PyMatching (MWPM decoding), and Sinter
(large-scale threshold estimation) to simulate and decode a repetition code
under a phenomenological noise model.
"""


# ----------------------------------------------------------------------
# # Installing Python Packages
#  I'm installing essential Python packages in my workspace.
# ----------------------------------------------------------------------

# !pip install stim==1.14  # run this in your terminal / environment before executing the script
# !pip install numpy  # run this in your terminal / environment before executing the script


# ----------------------------------------------------------------------
# # Checking Versions
# I am checking the version of the above-installed Python package, Stim.
# ----------------------------------------------------------------------

import stim
print(stim.__version__)


# ----------------------------------------------------------------------
# # Generating Circuit
# I'm going to generate a quantum error-correcting circuit called "Repetition Circuit". For this purpose, I am using a code distance d=9 that will generate a circuit of a total of 17 qubits (indexed from 0 through 16). This circuit has 9 data qubits (controlled qubits) and 8 ancilla (target qubits).
# ----------------------------------------------------------------------

circuit = stim.Circuit.generated(             # I generate circuit i.e repetition circuit
    "repetition_code:memory",                 # I store informaton in memory that protects it from errors
    rounds=25,                                # I run my circuit 25 times
    distance=9,                               # I generate my desired circuit
    before_round_data_depolarization=0.04,    # I introduce DEPOLARIZE1(0.04) with 4% chances of error in circuit
    before_measure_flip_probability=0.01)     # I insert X error by selecting its probability p=10*-3(0.01) of flipping quibits before measurements
print(repr(circuit))                          # I print the complete textual description of my generated repeititon circuit

circuit_diagram = circuit.diagram('timeline-svg')   # I generated my circuit diagram by using "SCalable VECTOR GRAPHICS (SVG)"
with open("circuit_diagram.svg", "w") as f:          # In a notebook this renders automatically; as a script we save it to a file instead.
    f.write(str(circuit_diagram))


# ----------------------------------------------------------------------
# ## Sampling The Circuit's Measurements This means I'm repeatedly running my circuit and recording the measurements each time to study its behavior and estimate its error rate. Each run is called a sample. Sampling is done because errors happen randomly, and by running it hundreds or thousands of times, it tells:
# 1. How often errors occur.
# 2. Whether the repetition code detects those errors.
# 3. How well the repetition code protects the stored quantum information.
# ----------------------------------------------------------------------

sampler = circuit.compile_sampler()                     # I am preparing my circuit to run so that it can generate measurements.
one_sample = sampler.sample(shots=1)[0]                 # I'm running the circuit once and saving the result of measurements.
for k in range(0, len(one_sample), 8):                  # I go through the result with step size of 8 at a time.
  timeslice = one_sample[k:k+8]                         # I take one group of 8 results. Each group corresponds to one round of measurements.
  print("".join("1" if e else "_" for e in timeslice))  # i make it to display results in terms of "1" and "_".


# ----------------------------------------------------------------------
# Here, in the output, we get two things:
# 1. _ which means no error occurred.
# 2. 1 which means an error occurred.
# When a data qubit gets an error (i.e., flips), it usually remains in the wrong state until the end of the circuit. Because of this, the neighboring stabilizer measurements keep reporting the same error in every round. That is why I get a long streak of 1s in the measurement results.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ## Sampling Circuit's Detectors
#  Instead of looking at raw measurements, we can look at the detectors. A detector doesn't report the same error repeatedly. It only reports when something changes (i.e., a detector fires), such as when an error first appears or disappears, making the data much easier to understand.
# ----------------------------------------------------------------------

detector_sampler = circuit.compile_detector_sampler()      # I am preparing my circuit to run so that it can generate detector's outcome.
one_sample = detector_sampler.sample(shots=1)[0]           # I'm running the circuit once and saving the result of detector.
for k in range(0, len(one_sample), 8):                     # I go through the result with step size of 8 at a time.
  timeslice = one_sample[k:k+8]                            # I take one group of 8 results. Each group corresponds to one round of measurements.
  print("".join("!" if e else "_" for e in timeslice))     # I make it to display results in terms of "!" and "_".



# ----------------------------------------------------------------------
# Now, instead of 1s, the detector detects '!' in pairs. This is because one error affects two nearby stabilizers. These pairs help the decoder to locate the error. At edges, a detector may pair with the boundary instead of another detector. The decoder uses this information to determine whether the logical qubit was flipped. If the logical qubit was flipped an odd number of times, it is corrected. If it is flipped an even number of times, no error correction is needed.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ## Using Detector Error Model and PyMatching
#  Stim makes it easier to use a decoder by converting a circuit into a "Detector Error Model (DEM)". A detector error model is a list of all errors in a quantum circuit that tells how errors appear in measurements.
#
# Note: To view the detector error model as a graph, use 'matchgraph-svg'.
# ----------------------------------------------------------------------

dem = circuit.detector_error_model()           # I am asking stim to convert my circuit into DEM to analyze my circuit and tell me
                                               # which error can cause which detector to fire i.e. detects error.
print(repr(dem))                               # I am making my DEM to be represented textually.

dem_diagram = dem.diagram("matchgraph-svg")          # Making my DEM to be viewed as a weighted graph.
with open("dem_matchgraph.svg", "w") as f:            # In a notebook this renders automatically; as a script we save it to a file instead.
    f.write(str(dem_diagram))



# ----------------------------------------------------------------------
# In the detector error model, each node is a detector, and each edge represents a possible error. The decoder matches pairs of detector events to find the most likely errors.
#
# A detector error model (DEM) is easier for a decoder to use than the original quantum circuit because the decoder represents the circuit as a weighted graph, which makes error correction faster and simpler.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# #### **Role of PyMatching**
# Now I am going to use PyMatching as my decoder. PyMatching is the Minimum-Weight Perfect Matching (MWPM) decoder written by Oscar Higgott. It uses detector events from Stim, builds a weighted graph from the DEM, and uses the MWPM algorithm to find and correct the most likely pattern of errors.
# ----------------------------------------------------------------------

# !pip install pymatching           # Importing and installing PyMatching package to find and correct possible errors in my workspace.  # run this in your terminal / environment before executing the script
import pymatching
import numpy as np                # Importing above intalled numpy package as "np".
print(pymatching.__version__)     # Printing their versions.
print(np.__version__)

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:    # I'm defining a count logical error function that counts how many
                                                                           # logical errors occurafter running my generated circuit num_shots times.

    sampler = circuit.compile_detector_sampler()                           # preparing my circuit to run repeatedly and produce detection events intead of raw measurements.
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)  # It runs my circuit num_shots times and tells two things each time:
                                                                                               # 1. which detector reports an error 2. whether the logical quibit was flipped(True)
                                                                                               # or not(False)

    detector_error_model = circuit.detector_error_model(decompose_errors=True)     #creating DEM that will create a map and tells our decoder which detector fires when error happens.
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)  # I create a PyMatching(MWPM) decoder that uses the error map created by DEM.

    predictions = matcher.decode_batch(detection_events)          # Here my decoder analyzes all detection events and predicts where logical quibit was flipped.

    num_errors = 0                                                # I initialize logical errors to be counted from zero.
    for shot in range(num_shots):                                 # I'm checking each shot one by one using for loop.
       actual_for_shot = observable_flips[shot]                   # Checking whether the logical quibit really flipped or not.
       predicted_for_shot = predictions[shot]                     # Getting the decoder's prediction.
       if not np.array_equal(actual_for_shot, predicted_for_shot):# I'm comparing the actual result with decoder's prediction to check whether it was correct or not.
        num_errors +=1                                            # If the decoder was wrong, increase the error count by 1.
    return num_errors                                             # It returns the total number of logical errors including decoder mistakes.


# ----------------------------------------------------------------------
# Now, I'm running the circuit, which was constructed earlier in this workspace with distance d=9, rounds=25, and a depolarization probability p=0.04 (or 4%). I am executing 100,000 shots, which is done in milliseconds.
# ----------------------------------------------------------------------

# %time  # (Jupyter magic — remove/replace with `import time` timing if running as plain script)
num_shots= 100_000                                                # I'm running 100_000 times my circuit to check how many times
num_logical_errors = count_logical_errors(circuit, num_shots)     # decoder make wrong predictions.
print("There were", num_logical_errors, "wrong predictions(logical errors) out of", num_shots, "shots")


# ----------------------------------------------------------------------
# As increasing 'physical noise strength' increases the logical error rate, we can check this by increasing the depolarization probability to 0.13 (or 13%).
# ----------------------------------------------------------------------

circuit = stim.Circuit.generated(                     # I'm checking the same circuit using same conditions(as we did in previous section)
                                                      #and only change depolarization probability to p=13% or 0.13
    "repetition_code:memory",
    distance=9,
    rounds=25,
    before_round_data_depolarization=0.13,
    before_measure_flip_probability=0.01)
num_shots = 100_000
num_logical_errors = count_logical_errors(circuit,num_shots)
print("There were", num_logical_errors, "wrong predictions(logical errors) out of", num_shots, "shots")


# ----------------------------------------------------------------------
# Now, I'm calculating the logical error rate. It should be around 1.5e-3.
# ----------------------------------------------------------------------

logical_error_rate = num_logical_errors/num_shots             # I'm finding logical error rate.
logical_error_rate


# ----------------------------------------------------------------------
# ## Estimating The Threshold of a Repetition Code:
# * Here I will test an error-correcting repetition code with different physical error rates and different code distances.
# * Then I will use **"Monte Carlo Sampling"** to simulate many noisy error runs.
# * Monte Carlo Sampling is a computational method that estimates a result by performing many random simulations.
# * In this context, it is used to repeat the same experiment many times with randomly generated errors to estimate the probability of failure.
# * The logical error rate calculated above is then used to plot **Logical Error Rate vs. Physical Error Rate** for each code distance.
# * The point where the curves on the plot intersect is the **threshold of our error-correcting repetition code.**
# * **Below the threshold:** Increasing the code distance reduces logical errors.
# * **Above the threshold:** Increasing the code distance no longer helps and may even increase logical errors.
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
num_shots = 100_000
for d in [3, 5, 7]:
    xs = []
    ys = []
    for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
      circuit = stim.Circuit.generated(
         "repetition_code:memory",
         distance = d,
         rounds = d*3,
         before_round_data_depolarization = noise)
      num_error_sampled = count_logical_errors(circuit, num_shots)
      xs.append(noise)
      ys.append(num_error_sampled/num_shots)
    plt.plot(xs, ys, label="d"+str(d))
plt.loglog()
plt.xlabel("Pyhsical Error Rate")
plt.ylabel("Logical Error Rate Per Shot")
plt.legend()
plt.show()


# ----------------------------------------------------------------------
# * The repetition code performs well in these simulations because the results are better than expected, as I used a simple noise model instead of a realistic circuit-level noise model.
# * My simulation uses depolarizing errors. Also, the repetition code is not affected by Z errors.
# * Since depolarizing error is a Z error, many **errors don't affect the code** because in a **depolarizing error, 1 out of every 3 errors is a Z error.**
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ## Using Sinter
#  Sinter is a simulation tool that works with Stim to automate large-scale error correction experiments. While Stim only simulates a single quantum circuit, Sinter manages and analyzes many Stim simulations automatically.
# ##### **Features of Sinter**
# * Runs many Stim simulations automatically.
# * Tests multiple code distances and physical error rates.
# * Collects logical error statistics.
# * Supports Monte Carlo Sampling.
# * Helps estimate the threshold and generate error-rate plots.
# ----------------------------------------------------------------------

# !pip install sinter~=1.14  # run this in your terminal / environment before executing the script
import sinter
from typing import List


tasks = [
    sinter.Task(
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance = d,
            rounds = d*3,
            before_round_data_depolarization=noise,
            ),
        json_metadata={'d': d, 'p': noise},
    )
    for d in [3, 5, 7, 9]
    for noise in [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
]
collected_stats: List[sinter.TaskStats]= sinter.collect(
    num_workers = 4,
    tasks = tasks,
    decoders = ['pymatching'],
    max_shots = 100_000,
    max_errors = 500,
)


# ----------------------------------------------------------------------
# I'm now going to make a plot by collecting data from Sinter and creating a professional graph of physical error rate vs. logical error rate per shot. It automatically highlights the region of uncertainty in the estimates.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(1,1)                                # I'm now going to make a graph inside one figure.
sinter.plot_error_rate(                                    # I'm drawing logical error rate plot using collected data from sinter.
    ax = ax,                                               # telling to draw graph on this axis
    stats = collected_stats,                               # Using statistics result of sinter calculated above.
    x_func = lambda stats: stats.json_metadata['p'],       # I'm choosing physical error rate 'p' on horizontal axis.

    group_func = lambda stats: stats.json_metadata['d'],   # choosing different code distances 'd' for drawing separate curve on each 'd'
                                                           # where lambda is a tiny function that takes results calculated earlier and
                                                           # plot it on axis
  )
ax.set_ylim(1e-4, 1e-0)                                    # I'm setting my graph limits.
ax.set_xlim(5e-2, 5e-1)
ax.loglog()
ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
ax.set_xlabel("Pyhsical Error Rate")
ax.set_ylabel("Logical Error Rate Per Shot")
ax.grid(which='major')                                     #
ax.grid(which='minor')
ax.legend()
fig.set_dpi(120)                                           # I make the graph resolution bigger by increasing Dots Per Inch(DPI).
fig.savefig("threshold_plot.png", bbox_inches="tight")     # Saving the final plot to a file (a script has no notebook cell to auto-display it in).
plt.show()


