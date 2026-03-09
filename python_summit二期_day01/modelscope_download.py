from modelscope import snapshot_download

# 模型名字
name = 'qwen/Qwen2.5-7B-Instruct-GGUF'
# 模型存放路径，需要手动创建对应的目录，并保证有足够的空间，否则下载出错。
model_path = r'C:\huggingface'
model_dir = snapshot_download(
    name,   # 仓库中的模型名
    cache_dir=model_path,  # 本地保存路径，
    revision='master',   # 分支版本
    allow_file_pattern="Qwen2.5-7B-Instruct.Q5_0*.gguf" # 模糊匹配的文件名
)