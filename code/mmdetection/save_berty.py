from transformers import BertConfig, BertModel, AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

save_path = "./bert-base-uncased"
config.save_pretrained(save_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)