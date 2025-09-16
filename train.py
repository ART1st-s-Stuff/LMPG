from tasks.basic_language import t1_word_identification

t1_word_identification.set_seed(114514)
packed_models = t1_word_identification.load_checkpoint("checkpoints/word_identification.pth")
t1_word_identification.train(*packed_models, t1_word_identification.ds["train"])