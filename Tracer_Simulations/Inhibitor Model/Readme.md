This folder contains the codes for the Transcription Factor-Inhibitor Model.
'Promoter Activity' is a code that provides the output concentration of transcribed mRNA. This is required for setting the threshold for ligand sensing by varying the concentration of any component in the system.
The model is as follows: Ligand binds to the SynNotch and activates it. The Transcription Factor is released that enter the nucleus to bind to the Operator and begin Transcription. There is a Basal Level of Inhibitor produced in the system. The binding of the Inhibitor to the Transcription Factor prevents the binding of the Transcription Factor to the Promoter sequence. The concentration of the Inhibitor can be varied by varying its promoter (weak->strong). 

'Syn->TF+I->Pro' allows to visualize the concentrations of the different nodes of the circuit at different input ligand conditions (pulse<>Continuous<>Window). Sobol Analysis also performed to check dependencies on parameters.

'Syn->TF+I->Pro Exponential Ligand' allows to visualize the concentrations of Reporter at exponential fold changes of ligand input.

'Syn->TF+I->Pro Linear Ligand Input' allows to visualize the concentrations of Reporter at linear fold changes of ligand input.
