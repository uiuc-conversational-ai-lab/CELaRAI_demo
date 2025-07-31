from datasets import disable_caching, load_dataset

disable_caching()
dataset = load_dataset("Thomas-X-Yang/celarai_demo", name="ingested")
print(dataset["train"][0])