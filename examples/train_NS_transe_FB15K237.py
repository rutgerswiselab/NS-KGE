from openke.config import Trainer, Tester
from openke.module.model import NS_TransE
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/",
	nbatches = 1,
	# batch_size = 1000,
	threads = 8, 
	sampling_mode = "WholeSampling",
	bern_flag = 1, 
	filter_flag = 1, 
	# neg_ent = 25,
    neg_ent = 0,
	neg_rel = 0,
	head_batch_size=128,
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = NS_TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	p_norm = 1, 
	norm_flag = True,
	pos_para = 1,
	neg_para = 0.0001,
)

# train the model
trainer = Trainer(model = transe, data_loader = train_dataloader, train_times = 2000, alpha = 0.0001, use_gpu = True, opt_method='adam', weight_decay=0.1, lr_decay=0.7)
trainer.run()

# test the model
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True, trainer = trainer)
tester.run_link_prediction(type_constrain = False)
