from pycrawlers import huggingface

# 你的token
token = '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'
# 实例化类
hg = huggingface(token=token)
# url = 'https://hf-mirror.com/google-bert/bert-base-cased/tree/main'
url = 'https://hf-mirror.com/sentence-transformers/paraphrase-mpnet-base-v2/tree/main'
# url = 'https://hf-mirror.com/sentence-transformers/paraphrase-mpnet-base-v2/tree/main'
# url = 'https://hf-mirror.com/sentence-transformers/all-roberta-large-v1/tree/main'

# 单个下载
# 默认保存位置在当前脚本所在文件夹 ./
# hg.get_data(url)

# 自定义下载位置
path = './sentence-transformers/paraphrase-mpnet-base-v2'
hg.get_data(url, path)