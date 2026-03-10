import numpy as np
import os
from lib.utils.log_utils import args_to_string
from lib.utils.sr_utils import sr_families
import wandb

# for evaluation

def extract_results_molecular_datasets(args, results, result_folder):
    # Extract results
    train_curves = [curves['train'] for curves in results]
    val_curves = [curves['val'] for curves in results]
    test_curves = [curves['test'] for curves in results]
    best_idx = [curves['best'] for curves in results]
    last_train = [curves['last_train'] for curves in results]
    last_val = [curves['last_val'] for curves in results]
    last_test = [curves['last_test'] for curves in results]

    # Extract results at the best validation epoch.
    best_epoch_train_results = [train_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_train_results = np.array(best_epoch_train_results, dtype=float)
    best_epoch_val_results = [val_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_val_results = np.array(best_epoch_val_results, dtype=float)
    best_epoch_test_results = [test_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_test_results = np.array(best_epoch_test_results, dtype=float)

    # Compute stats for the best validation epoch
    mean_train_perf = np.mean(best_epoch_train_results)
    std_train_perf = np.std(best_epoch_train_results, ddof=1)  # ddof=1 makes the estimator unbiased
    mean_val_perf = np.mean(best_epoch_val_results)
    std_val_perf = np.std(best_epoch_val_results, ddof=1)  # ddof=1 makes the estimator unbiased
    mean_test_perf = np.mean(best_epoch_test_results)
    std_test_perf = np.std(best_epoch_test_results, ddof=1)  # ddof=1 makes the estimator unbiased
    min_perf = np.min(best_epoch_test_results)
    max_perf = np.max(best_epoch_test_results)

    # Compute stats for the last epoch
    mean_final_train_perf = np.mean(last_train)
    std_final_train_perf = np.std(last_train, ddof=1)
    mean_final_val_perf = np.mean(last_val)
    std_final_val_perf = np.std(last_val, ddof=1)
    mean_final_test_perf = np.mean(last_test)
    std_final_test_perf = np.std(last_test, ddof=1)
    final_test_min = np.min(last_test)
    final_test_max = np.max(last_test)

    msg = (
        f"========= Final result ==========\n"
        f'Dataset:                {args.dataset}\n'
        f'SHA:                    {args.sha}\n'
        f'----------- Best epoch ----------\n'
        f'Train:                  {mean_train_perf} ± {std_train_perf}\n'
        f'Valid:                  {mean_val_perf} ± {std_val_perf}\n'
        f'Test:                   {mean_test_perf} ± {std_test_perf}\n'
        f'Test Min:               {min_perf}\n'
        f'Test Max:               {max_perf}\n'
        f'----------- Last epoch ----------\n'
        f'Train:                  {mean_final_train_perf} ± {std_final_train_perf}\n'
        f'Valid:                  {mean_final_val_perf} ± {std_final_val_perf}\n'
        f'Test:                   {mean_final_test_perf} ± {std_final_test_perf}\n'
        f'Test Min:               {final_test_min}\n'
        f'Test Max:               {final_test_max}\n'
        f'---------------------------------\n\n')
    print(msg)

    

    if (not args.debug):
        wandb.log({
            'Mean Train Performance': mean_train_perf,
            'Mean Val Performance': mean_val_perf,
            'Mean Test Performance': mean_test_perf
        })
        # additionally write msg and configuration on file
        msg += args_to_string(args)
        filename = os.path.join(result_folder, "results.txt")
        print('Writing results at: {}'.format(filename))
        with open(filename, 'w') as handle:
            handle.write(msg)
        wandb.save(filename)
    
def extract_results_tu_datasets(args, results, result_folder):
    # aggregate results
    val_curves = np.asarray([curves['val'] for curves in results])
    avg_val_curve = val_curves.mean(axis=0)
    best_index = np.argmax(avg_val_curve)
    mean_perf = avg_val_curve[best_index]
    std_perf = val_curves.std(axis=0)[best_index]

    print(" ===== Mean performance per fold ======")
    perf_per_fold = val_curves.mean(1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Max performance per fold ======")
    perf_per_fold = np.max(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Median performance per fold ======")
    perf_per_fold = np.median(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Performance on best epoch ======")
    perf_per_fold = val_curves[:, best_index]
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Final result ======")
    msg = (
        f'Dataset:        {args.dataset}\n'
        f'Accuracy:       {mean_perf} ± {std_perf}\n'
        f'Best epoch:     {best_index}\n'
        '-------------------------------\n')
    print(msg)

    if not args.debug:
        wandb.log({
            'Accuracy': mean_perf
        })
        
        # additionally write msg and configuration on file
        msg += args_to_string(args)
        filename = os.path.join(result_folder, 'result.txt')
        print('Writing results at: {}'.format(filename))
        with open(filename, 'w') as handle:
            handle.write(msg)
        wandb.save(filename)

def extract_results_sr_datasets(args, results, result_folder):
    families = sr_families()
    msg = (
        f"===== Final result ======\n"
        f'Datasets:\tSR-GRAPHS\n')
    for i, f in enumerate(families):
        curves = results[i]
        test_perfs = [curve['last_test'] for curve in curves]
        assert len(test_perfs) == args.stop_seed + 1 - args.start_seed
        mean = np.mean(test_perfs)
        std_err = np.std(test_perfs) / float(len(test_perfs))
        minim = np.min(test_perfs)
        maxim = np.max(test_perfs)
        msg += (
            f'------------------ {f} ------------------\n'
            f'Mean failure rate:     {mean}\n'
            f'StdErr failure rate:   {std_err}\n'
            f'Min failure rate:      {minim}\n'
            f'Max failure rate:      {maxim}\n'
            '-----------------------------------------------\n')
    print(msg)

    if not args.debug:
        # additionally write msg and configuration on file
        msg += args_to_string(args)
        filename = os.path.join(result_folder, 'result.txt')
        print('Writing results at: {}'.format(filename))
        with open(filename, 'w') as handle:
            handle.write(msg)
        wandb.save(filename)


import numpy as np
import os
import wandb
from lib.utils.log_utils import args_to_string

def extract_results_cosc_graphs(args, results, result_folder):
    """
    results: list of dicts (one per run/seed/fold), where each dict is:
      {
        'train': [.. per-epoch ..],
        'val':   [.. per-epoch ..],
        'test':  [.. per-epoch ..],
        'last_train': float,
        'last_val': float,
        'last_test': float,
        'best': int,   # best epoch index into curves
      }
    """

    train_curves = [r["train"] for r in results]
    val_curves   = [r["val"]   for r in results]
    test_curves  = [r["test"]  for r in results]
    best_idx     = [r["best"]  for r in results]

    last_train = np.asarray([r["last_train"] for r in results], dtype=float)
    last_val   = np.asarray([r["last_val"]   for r in results], dtype=float)
    last_test  = np.asarray([r["last_test"]  for r in results], dtype=float)

    # Best-epoch metrics per run (based on that run's best val epoch)
    best_train = np.asarray([train_curves[i][b] for i, b in enumerate(best_idx)], dtype=float)
    best_val   = np.asarray([val_curves[i][b]   for i, b in enumerate(best_idx)], dtype=float)
    best_test  = np.asarray([test_curves[i][b]  for i, b in enumerate(best_idx)], dtype=float)

    # Stats helper (ddof=1 only if n>1 to avoid warnings / nan)
    def mean_std(x):
        x = np.asarray(x, dtype=float)
        mean = float(np.mean(x))
        std  = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        return mean, std, float(np.min(x)), float(np.max(x))

    mean_bt, std_bt, _, _ = mean_std(best_train)
    mean_bv, std_bv, _, _ = mean_std(best_val)
    mean_bx, std_bx, min_bx, max_bx = mean_std(best_test)

    mean_lt, std_lt, _, _ = mean_std(last_train)
    mean_lv, std_lv, _, _ = mean_std(last_val)
    mean_lx, std_lx, min_lx, max_lx = mean_std(last_test)

    metric_name = getattr(args, "eval_metric", "Metric")
    msg = (
        f"========= Final result ==========\n"
        f"Dataset:                {args.dataset}\n"
        f"SHA:                    {getattr(args, 'sha', 'N/A')}\n"
        f"Runs:                   {len(results)}\n"
        f"Metric:                 {metric_name}\n"
        f"----------- Best epoch (per-run best val) ----------\n"
        f"Train:                  {mean_bt} ± {std_bt}\n"
        f"Valid:                  {mean_bv} ± {std_bv}\n"
        f"Test:                   {mean_bx} ± {std_bx}\n"
        f"Test Min:               {min_bx}\n"
        f"Test Max:               {max_bx}\n"
        f"----------- Last epoch ----------\n"
        f"Train:                  {mean_lt} ± {std_lt}\n"
        f"Valid:                  {mean_lv} ± {std_lv}\n"
        f"Test:                   {mean_lx} ± {std_lx}\n"
        f"Test Min:               {min_lx}\n"
        f"Test Max:               {max_lx}\n"
        f"---------------------------------\n\n"
    )
    print(msg)

    if not args.debug:
        wandb.log({
            f"Best/Train {metric_name} (mean)": mean_bt,
            f"Best/Val {metric_name} (mean)":   mean_bv,
            f"Best/Test {metric_name} (mean)":  mean_bx,
            f"Last/Train {metric_name} (mean)": mean_lt,
            f"Last/Val {metric_name} (mean)":   mean_lv,
            f"Last/Test {metric_name} (mean)":  mean_lx,
        })

        msg_out = msg + args_to_string(args)
        filename = os.path.join(result_folder, "results.txt")
        print(f"Writing results at: {filename}")
        with open(filename, "w") as handle:
            handle.write(msg_out)
        wandb.save(filename)



def print_summary(summary):
    msg = ''
    for k, v in summary.items():
        msg += f'Fold {k:1d}:  {v:.3f}\n'
    print(msg)
