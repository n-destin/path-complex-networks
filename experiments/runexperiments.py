import os
import subprocess
import time
from collections import deque

scripts_path = os.path.join("experiments", "newscripts")
command = os.path.join(scripts_path, "experiments.sh")

normal_graph_models = ["graphsage", "GAT", "GCN", "graphTransformer", "graphIsomorphism"]
complex_types = ["path", "cell", "simplicial"]
ogb_node_prediction_datasets = ["PRODUCTS", "PROTEINS", "ARXIV", "PAPERS100M", "MAG"]
ogb_link_prediction_datasets = ["PPA", "COLLAB", "DDI", "CITATION2", "WIKIKG2", "BIOKG", "VESSEL"]

GPUS = ["0", "1", "2"]  # your 3 GPUs


def make_jobs():
    jobs = []

    # for dataset in ogb_node_prediction_datasets:
    #     jobs.append((dataset, "space_cin", "path", "node"))

    # for dataset in ogb_link_prediction_datasets:
    #     jobs.append((dataset, "space_cin", "path", "pair"))

    # for complex_type in complex_types:
    #     jobs.append(("cosc-structural-graphs", "sparse_cin", complex_type, "pair"))

    for model in normal_graph_models:
        # jobs.append(("cosc-structural-graphs", model, "", "pair"))
        for dataset_ in ogb_node_prediction_datasets:
            jobs.append((dataset_, model, "", "node"))
            break
        for dataset_ in ogb_link_prediction_datasets:
            jobs.append((dataset_, model, "", "pair"))
            break

    return jobs


def launch_job(gpu, dataset, model, complex_type, task):
    args = [
        "bash",
        command,
        dataset,
        model,
        gpu,
        complex_type,
        task,
    ]
    print(f"Launching on GPU {gpu}: dataset={dataset}, model={model}, complex_type={complex_type}, task={task}")
    return subprocess.Popen(args)


def main():
    job_queue = deque(make_jobs())
    running = {}  # gpu -> (process, job)

    while job_queue or running:
        # fill free GPUs
        for gpu in GPUS:
            if gpu not in running and job_queue:
                job = job_queue.popleft()
                proc = launch_job(gpu, *job)
                running[gpu] = (proc, job)

        # check finished jobs
        finished_gpus = []
        for gpu, (proc, job) in running.items():
            ret = proc.poll()
            if ret is not None:
                print(f"Finished on GPU {gpu} with code {ret}: {job}")
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del running[gpu]

        time.sleep(5)


if __name__ == "__main__":
    main()