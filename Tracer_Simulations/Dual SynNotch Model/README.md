This folder contains the codes for the Dual SynNotch Model.
The model is as follows: The ligand is sensed by a low-affinity SynNotch that is activated only at high concentrations of ligand. This promotes the transcription of a high affinity SynNotch that detects ligand at lower concentrations. The low affinity SynNotch allows for thresholding of the ligand detection while the cascade allows for removing leaky signals due to noise.

'Syn->TF1->TF2->Pro' allows for visualizing the concentrations of the various nodes of the circuit.

'Syn->TF1->TF2->Pro Different Promoters' does the same for different promoters that can be used to produce the second SynNotch.

'Syn->TF1->TF2+Degron->Pro' does the same for when we attach different degrons to the second SynNotch molecule
