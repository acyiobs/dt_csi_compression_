## Digital Twin Aided Massive MIMO: CSI Compression and Feedback


The implementation for the simulations of the paper "[Digital Twin Aided Massive MIMO: CSI Compression and Feedback](https://arxiv.org/html/2402.19434v1)".

### Abstract

*Deep learning (DL) approaches have demonstrated high performance in compressing and reconstructing the channel state information (CSI) and reducing the CSI feedback overhead in massive MIMO systems. One key challenge, however, with the DL approaches is the demand for extensive training data. Collecting this real-world CSI data incurs significant overhead that hinders the DL approaches from scaling to a large number of communication sites. To address this challenge, we propose a novel direction that utilizes site-specific digital twins to aid the training of DL models. The proposed digital twin approach generates site-specific synthetic CSI data from the EM 3D model and ray tracing, which can then be used to train the DL model without real-world data collection. To further improve the performance, we adopt online data selection to refine the DL model training with a small real-world CSI dataset. Results show that a DL model trained solely on the digital twin data can achieve high performance when tested in a real-world deployment. Further, leveraging domain adaptation techniques, the proposed approach requires orders of magnitude less real-world data to approach the same performance of the model trained completely on a real-world CSI dataset.*

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

If you in any way use this code for research that results in publications, please cite our original article:

> S. Jiang and A. Alkhateeb, "Digital Twin Aided Massive MIMO: CSI Compression and Feedback." arXiv preprint arXiv:2402.19434, 2024.
