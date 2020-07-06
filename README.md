# Using reinforcement learning to solve VRP

This repository provides an implementation of a Capacitated Vehicle Routing Problem (CVRP) solver by using the Operations Research Tools (OR-Tools) [1] and the Reinforcement Learning (RL) framework proposed by Kool et. al (2019). The RL implementation is based on the code available in [2], which is the implementation in Python provided by Kool et. al (2019) of the RL framework proposed in [3].

This work was done in the context of a machine learning course of the  Master's Degree in Data Science at the University of Lisbon.

The file report.pdf contains the report explaining how the reinforcement learning framework works and the work done to compare it with the OR-Tools heuristics. The code produced in notebook.ipynb generates random CVRP problems and solves them by using both OR-Tools and Kool et. al (2019) RL framework. 

I obtained different results than the ones reported in [3], especially for smaller problem sizes, probably because the authors changed code from the CVRP solver of the OR-Tools to compare it with their framework. For more details about the reinforcement learning framework proposed see [2,3].

# References

[1] https://developers.google.com/optimization

[2] https://github.com/wouterkool/attention-learn-to-route
    
[3]  Kool, W., Hoof, H.V., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR.



