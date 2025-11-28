# To make it work
.venv/lib/python3.13/site-packages/transformers/trainer.py, line 4154
```
loss = loss * (self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)
```