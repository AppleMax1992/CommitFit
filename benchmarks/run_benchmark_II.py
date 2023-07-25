from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments


# batch_sizes List of batch sizes for which memory and time performance will be evaluated
# sequence_lengths List of sequence lengths for which memory and time performance will be evaluated
# save_to_csv Save result to a CSV file
# env_print Whether to print environment information
args = PyTorchBenchmarkArguments(models=['O-2-best','O-all-2-best'], batch_sizes=[16], sequence_lengths=[8, 32, 128, 512], save_to_csv=True,env_print=True)


benchmark = PyTorchBenchmark(args)

results = benchmark.run()
print(results)