# Teknofest2026_code
This repository contains all the ODEs and codes for the synthetic circuit designed for participating in Teknofest 2026 as a part of Team BioEnGene, IIT KGP. We worked on two models- Inhibitor Model and Dual SynNotch Model.
Both the circuits were initialized by SynNotch molecules that were constitutive expressed and present on the cell's surface. The SynNotch was activated by CD19 and the Transcription Factor tTA was released.

In the Inhibitor Model, the Transcription Factor tTA would be bound by a doxycycline mimicing peptide sequence that is constitutively produced and prevent the binding of tTA to the Promoter. This will result in the formation of an inactive complex that allows us to set a threshold for ligand sensing.

In the Dual SynNotch Model, the first SynNotch molecule that has a lower binding affinity towards CD19 and will get activated only in the presence of a larger concentration of CD19. The Transcription Factor will tandemly express a second SynNotch molecule that has a higher binding affinity towards CD19. This will release a second Transcription Factor that will produce the reporter.
