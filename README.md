# Run
<!-- .venv/lib/python3.13/site-packages/transformers/trainer.py, line 4154
```
loss = loss * (self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)
``` -->

1. Install transformers==4.53.0, trl==0.20.0, rwkv-fla (use their github repo link), and other missing packages.
2. Download ARWKV-7B-Preview-0.1 model into models/arwkv.
3. Disable dynamo by manually editing `models/arwkv/wkv.py`
    ```
    if check_pytorch_version("2.6"):
        compile_decorator = torch.compiler.disable
        torch._dynamo.config.cache_size_limit = 512
    #else:
        # def compile_decorator(func):
        #     return func
    ```
4. `python run/stage1-sft-arwkv.py`
5. `python stage0-eval.py`
    > args:
    > ```
    > --model <qwen|arwkv|tuned>     Tuned: arwkv-tuned
    > --type <number|random|calc>    How <content> is generated.
    > --insert-trash                 If to insert long irrelavent paragraph.
    > ```