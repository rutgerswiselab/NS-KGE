from openke.config import Trainer, Tester
from openke.module.model import NS_Simple
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
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
simple = NS_Simple(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	pos_para = 1,
	neg_para = 0.0001,
)


# train the model
trainer = Trainer(model = simple, data_loader = train_dataloader, train_times = 2000, alpha = 0.0001, use_gpu = True, on_step=True, opt_method='adam')
trainer.run()

# test the model
tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = True, trainer = trainer)
tester.run_link_prediction(type_constrain = False)
