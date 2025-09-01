This is a repository to recreate the results of TriMoE.

First, you will need to gain access to the MIMIC-III dataset on PhysioNet:

https://physionet.org/content/mimiciii/1.4/

Then you have to run the Harutyunyan et al (2017) 48 hour in hospital mortality benchmark. This takes roughly 10 hours on a virtual google machine and can be accessed here:

https://github.com/YerevaNN/mimic3-benchmarks

Following that you must run the Khadanga et al (2019) T0 file, available here:

https://github.com/kaggarwal/ClinicalNotesICU

Finally run the preprocessing file used by Han et al (2024) (developed by Zhang et al (2023):

https://github.com/aaronhan223/FuseMoE/tree/main/src/preprocessing

Now you can run the preprocessing script stored in this paper by downloading the preprocessing file in this repository and running run_pipeline.ssh

Following that, you can run the model - the Slurm script offers a way to adjust the hyperparameters. Set to the exact same values as stored in hyperparamter.png in this file, or the file in the appendix.
