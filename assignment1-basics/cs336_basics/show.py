import pickle

with open("data/tinystories_bpe.pkl", "rb") as f:
    saved = pickle.load(f)

vocab = saved["vocab"]
merges = saved["merges"]

print("词表大小：", len(vocab))
print("合并规则数量：", len(merges))

for token_id in range(256, 1000):
    print(token_id, repr(vocab[token_id]))

