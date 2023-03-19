from Algo.arguments import parser
from Algo.dmc import train as rl_train
from Algo.dmc_self_play import train as rl_train_self_play
import os

if __name__ == "__main__":
	train_type = 'rl'
	os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
	if train_type == 'rl':
		flags = parser.parse_args()
		rl_train_self_play(flags)
		# rl_train(flags)
