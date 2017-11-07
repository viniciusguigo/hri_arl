### test_ddpg2
using conv nets to handle depth sensor, no imu states
10000 episodes
+1 reward going forward
+/-100 for completing or crashing
1 conv layers (32,3), 2 fc (400, 300 neurons each)
no random action, but using noise on actions
batch_size = 1
mem buffer = 10^4
total runtime: ~6hours (maybe, not sure)
conclusion: stuck going backwards

### test_ddpg3
using conv nets to handle depth sensor, no imu states
500 episodes
+20 reward going forward
+/-100 for completing or crashing
3 conv layers (32,3), 2 fc (200 neurons each)
.1 random actions, and noise on actions
batch_size = 16
mem buffer = 10^4
total runtime: n/a
conclusion: random behavior, maybe need more runs

### test_ddpg4
using conv nets to handle depth sensor, no imu states
2000 episodes
500 steps max
+/-20 reward per meter
+/-100 for completing or crashing
3 conv layers (32,3), 2 fc (200 neurons each)
.1 random actions, and noise on actions
batch_size = 16
mem buffer = 10^4
total runtime: 546min
conclusion: not able to learn, still random behavior

### test_random5
using conv nets to handle depth sensor, no imu states
2000 episodes
500 steps max
random agent
+/-20 reward per meter
total runtime: 222min
conclusion: random

### test_human5
using conv nets to handle depth sensor, no imu states
100 episodes
500 steps max
human agent
+/-20 reward per meter
+/-100 for completing or crashing
total runtime:
conclusion: human demonstration

### test_ddpg_small
using conv nets to handle depth sensor, no imu states
1000 episodes
500 steps max
+/-20 reward per meter
+/-100 for completing or crashing
3 conv layers (32,3), 2 fc (200 neurons each)
.1 random actions, and noise on actions
small actor and critic networks
batch_size = 16
mem buffer = 10^4
total runtime: n/a
conclusion: didnt learn more than before, net probably usntable

### test_ddpg_multi_16bs_offline
using conv nets to handle depth sensor, no imu states
2000 episodes
500 steps max
+/-20 reward per meter
+/-100 for completing or crashing
3 conv layers (32,3), 2 fc (200 neurons each)
.1 random actions, and noise on actions
offline, update in the end of run
batch_size = 16
mem buffer = 10^4
total runtime: 571min
conclusion: sweet, learning!

### test_ddpg_multi_16bs_1eps_offline
using conv nets to handle depth sensor, no imu states
2000 episodes
500 steps max
+/-20 reward per meter
+/-100 for completing or crashing
3 conv layers (32,3), 2 fc (200 neurons each)
.1 random actions, and noise on actions
offline, update in the end of run
eps = .1
batch_size = 16
mem buffer = 10^4
total runtime:
conclusion:
