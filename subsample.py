import argparse
import glob
from os.path import join, basename
from random import choice, choices, seed

import osmnx as ox
from joblib import Parallel, delayed
from osmnx.io import load_graphml

import networkx as nx
from networkx.algorithms.shortest_paths import single_source_dijkstra

from loguru import logger

COLOR_MAP = {'start': 'r', 'finish': 'g'}
SIZE_MAP = {'start': 24, 'finish': 24}


def prepare_tasklist(input_dir, target_sample_size, max_workers, dist_min, dist_max, dist_step, max_defect_ratio):
    file_list = glob.glob(join(input_dir, '*_*.graphml'))
    graph_iter = Parallel(n_jobs=max_workers, backend='multiprocessing')(
        delayed(load_graphml)(filename) for filename in file_list
    )
    logger.info("Processing {} city networks", len(graph_iter))
    city_weights = {
        filename: len(g)
        for filename, g in zip(file_list, graph_iter)
    }
    weights_sum = sum(city_weights.values())
    city_probas = {filename: weight * 1.0 / weights_sum for filename, weight in city_weights.items()}
    node_dist_ranges = [(x, x + dist_step) for x in range(dist_min, dist_max, dist_step)]
    cities = list(city_probas.keys())
    max_sample_size = int((1 + max_defect_ratio) * target_sample_size / len(node_dist_ranges))
    networks_raw = choices(cities, weights=[city_probas[x] for x in cities], k=max_sample_size)
    networks, counter = list(), dict()
    for filename in networks_raw:
        if filename not in counter:
            counter[filename] = 0
        counter[filename] += 1
        networks.append((filename, counter[filename]))
    tasks = [
        (filename, split_seed, dist_from, dist_to)
        for filename, split_seed in networks
        for (dist_from, dist_to) in node_dist_ranges
    ]
    logger.info("Exploded {} networks into {} generation jobs", len(graph_iter), len(tasks))
    return tasks


def split_subgraph(g, split_seed, node_dist_range_from, node_dist_range_to, distance_coef=1.5):
    seed(split_seed)
    g_res = ox.get_digraph(g.copy(), 'travel_time')
    g_res.graph['split_seed'] = split_seed
    g_res.graph['distance_coef'] = distance_coef
    g_res.graph['node_distance_range_from'] = node_dist_range_from
    g_res.graph['node_distance_range_to'] = node_dist_range_to
    for u, v in g_res.edges:
        g_res[u][v]['travel_time'] = int(g_res[u][v]['travel_time'])
    node_from = choice([x for x in g_res])
    g_res.graph['start_node'] = node_from
    distances_from, _ = single_source_dijkstra(g_res, node_from, weight='travel_time')
    node_to_candidates = [node for node, dist in distances_from.items()
                          if node_dist_range_from <= dist < node_dist_range_to]
    if len(node_to_candidates) == 0:
        return None
    node_to = choice(node_to_candidates)
    g_res.graph['finish_node'] = node_to
    distances_to, _ = single_source_dijkstra(g_res.reverse(), node_to, weight='travel_time')
    nodes_remaining = [node for node in g_res
                       if node in distances_from and node in distances_to
                       and distances_from[node] + distances_to[node] <= distance_coef * distances_from[node_to]]
    g_res = nx.induced_subgraph(g_res, nodes_remaining)
    wcc = [x for x in nx.weakly_connected_components(g_res) if node_from in x and node_to in x]
    if len(wcc) != 1:
        return None
    (wcc,) = wcc
    return nx.induced_subgraph(g_res, wcc)


def store_graph(g, filename, split_seed, dist_from, dist_to):
    out_filename = basename(filename).replace('.graphml', f'_{dist_from}_{dist_to}_{split_seed}.graphml')
    out_filepath = join(args.output_dir, out_filename)
    ox.save_graphml(nx.MultiDiGraph(g), filepath=out_filepath)
    if args.plot_dir is not None:
        node_color = [COLOR_MAP.get(node.get('node_type'), 'k') for _, node in g.nodes(data=True)]
        node_size = [SIZE_MAP.get(node.get('node_type'), 3) for _, node in g.nodes(data=True)]
        ox.plot_graph(
            nx.MultiDiGraph(g), figsize=(16, 16),
            node_size=node_size, node_color=node_color,
            edge_linewidth=1, edge_color="#111", edge_alpha=.25, bgcolor="w",
            save=True, show=False, close=True,
            filepath=join(args.plot_dir, out_filename.replace('.graphml', '.png'))
        )


def process_task(filename, split_seed, dist_from, dist_to):
    g = load_graphml(filename)
    g = split_subgraph(g, split_seed, dist_from, dist_to)
    if g is not None:
        logger.info(
            "Job {}, split seed {}, distance range {}..{}, generating a graph with {} nodes and {} edges",
            filename, split_seed, dist_from, dist_to, len(g), len(g.edges)
        )
        store_graph(g, filename, split_seed, dist_from, dist_to)
    else:
        logger.warning(
            "Job {}, split seed {}, distance range {}..{}, skipping graph generation",
            filename, split_seed, dist_from, dist_to
        )


def main(args):
    logger.info("Reading the network sizes from {}", args.input_dir)
    tasklist = prepare_tasklist(args.input_dir, args.sample_size, args.max_workers,
                                args.dist_min, args.dist_max, args.dist_step, 0.01 * args.max_defect_pct)
    logger.info("Generated {} subsampling jobs", len(tasklist))
    Parallel(n_jobs=args.max_workers)(delayed(process_task)(*task_args) for task_args in tasklist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a subgraph shortest path problem instance dataset')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory with city GraphML files')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for instance GraphML files')
    parser.add_argument('--plot-dir', '-p', required=False, help='Output directory for instance map plots')
    parser.add_argument('--log-file', '-l', required=False, help='Write sampling log into the file')
    parser.add_argument('--sample-size', '-n', type=int, required=True,
                        help='Target number of generated problem instances')
    parser.add_argument('--dist-min', type=int, required=False, default=60,
                        help='Minimum travel time in seconds used for finish vertex sampling')
    parser.add_argument('--dist-max', type=int, required=False, default=300,
                        help='Maximum travel time in seconds used for finish vertex sampling')
    parser.add_argument('--dist-step', type=int, required=False, default=60,
                        help='Travel time range in seconds used for finish vertex sampling')
    parser.add_argument('--max-defect-pct', type=int, required=False, default=15,
                        help='Maximum percentage of instances that can be safely discarded')
    parser.add_argument('--max-workers', type=int, required=False, help='Maximum number of jobs run simultaneously')
    args = parser.parse_args()
    if args.log_file is not None:
        logger.add(args.log_file)
    main(args)
