This directory demos usage of some new features.

### `rnn_dropout`
under developing, progress=50%.

Trying to follow paper `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks` to implement a new dropout strategy.

- dropout mask is shared inside the same sequence. This can be achieved by using `share_dropout_mask_in_seq=True` in `addto_layer`
- droput mask is shared inside recurrent layers. This is ***not*** implemented

### `sampling_language_model`
under developing, progress=90%, unitests and gradient checks not implemented.

Using a newly implemented `sampled_fc_layer`, several different sampling strategies can be achieved. Check out `train.sh` for usage.

- Importance Sampling: use a Monte Carlo approximation of the negative phase. 
- Noise Contrastive Estimation: use a proxy problem to solve the optimization.
- Negative Sampling: used by Mikolov to train word2vec. Can be achieved by setting `nce=1,subtract_log_q=0`.
