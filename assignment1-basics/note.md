```bash
# 查看改动
git status
git diff

# 暂存当前目录下的全部改动
git add .

# 提交到本地仓库
git commit -m "Complete Embedding implementation and test adapter"

# 推送到远程仓库
git push
```



# 新增内容

1.train_bpe.py

2.fast_train_bpe.py

3.show.py

4.tokenizer.py

5.liner.py

6.linear.py







在数值表示上，一维的长度为d的张量与1*d的二维张量表示完全相同，但是二维的在计算时会进行广播所以注意二者的区别