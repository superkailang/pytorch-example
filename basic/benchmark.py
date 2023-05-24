import torch
import timeit
import torch.utils.benchmark as benchmark


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


def generate_benchmark(num_size=[100, 1000, 10000], num_threads=4):
    '''
        iter thread and size generate benchmark Measurement
        :returns List[common.Measurement]
    '''
    result = []
    label = "batch dot"
    for size in num_size:
        x = torch.randn(size, 64)
        for thread in range(num_threads):
            n_thread = thread + 1
            sub_layer = "{}_{}".format(size, n_thread)
            t0 = benchmark.Timer(
                stmt='batched_dot_mul_sum(x, x)',
                setup='from __main__ import batched_dot_mul_sum',
                num_threads=n_thread,
                label=label,
                sub_label=sub_layer,
                description='mul',
                globals={'x': x})

            t1 = benchmark.Timer(
                stmt='batched_dot_bmm(x, x)',
                setup='from __main__ import batched_dot_bmm',
                num_threads=n_thread,
                label=label,
                sub_label=sub_layer,
                description='bmm',
                globals={'x': x})
            result.append(t0.blocked_autorange())
            result.append(t1.blocked_autorange())
    return result


def main():
    # Input for benchmarking
    x = torch.randn(50000, 64)
    t0 = timeit.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals={'x': x})

    t1 = timeit.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals={'x': x})
    print(f'timeit mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
    print(f'timeit bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')

    num_threads = 6
    num_size = [100, 10000, 1000000]
    results = generate_benchmark(num_size, num_threads)
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()


if __name__ == '__main__':
    main()
