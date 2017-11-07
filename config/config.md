# CONFIG FILES

Save on this folder the config files for each experiment. Check this [example file](./example.ini).

## ICLR
**r_max = 10 per step**
iclr_prev_ddpg_0 (no clipping) - good
iclr_prev_ddpg_0_clip (clipping actions to +-.5) - bad
iclr_prev_ddpg_1 (clipping actions to +-.5) - good
iclr_prev_ddpg_2 (no clipping) -
iclr_prev_ddpg_1_noclip (no clipping) - so so
iclr_prev_ddpg_2_noclip (no clipping) - bad
iclr_prev_imitation (no clipping) - crashed (overflow)

iclr_prev_ddpg_2_v2 (no clipping), 5000 epi - started good, bad later
iclr_prev_ddpg_1_v2 (no clipping), 5000 epi - bad
iclr_prev_ddpg_1_v3 (clipping actions to +-.5), 5000 epi - maybe bad
iclr_prev_ddpg_0_v2 (no clipping), 5000 epi - bad
iclr_prev_ddpg_2_v3 (no clipping), 5000 epi - same bad
iclr_prev_ddpg_2_v4 (clipping actions to +-.5), 5000 epi - terrible

Using a bigger net for ddpg, added maxpool2d between conv layers
hri_ddpg_0_v0 (no clipping), 1000 epi -
hri_ddpg_1_v0 (no clipping), 1000 epi -
hri_ddpg_2_v0 (no clipping), 1000 epi -
