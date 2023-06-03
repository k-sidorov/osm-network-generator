import argparse
import csv
import glob
from joblib import Parallel, delayed
from os.path import join, basename

import osmnx as ox
from loguru import logger


def construct_graph(row, fallback_speed):
    try:
        logger.info('Processing row {}', row)
        query = f'{row["city"]}, {row["country"]}'
        try:
            g = ox.graph_from_place(query, network_type='drive')
            g = ox.add_edge_speeds(g, fallback=fallback_speed)
            g = ox.add_edge_travel_times(g)
        except ValueError as e:
            logger.warning('Error while extracting graph for query `{}`: {}', query, e)
            return False, (row['id'], query, e)
        for attr in ['id', 'city', 'country', 'iso2', 'iso3']:
            g.graph[attr] = row[attr]
        logger.info('Graph for row {} has been constructed with {} vertices and {} edges', row, len(g.nodes), len(g.edges))
        return True, g
    except Exception as e:
        logger.error('Unknown error encountered while processing row {}: {}', row, e)
        return False, (row['id'], None, e)


def store_graph(row, g, output_dir, plot_dir):
    filename = f'{row["id"]}_{row["city_ascii"]}_{row["iso2"]}.graphml'.replace(' ', '_')
    filepath = join(output_dir, filename)
    ox.save_graphml(g, filepath=filepath)
    if plot_dir is not None:
        ox.plot_graph(
            g, figsize=(16, 16),
            node_size=3, node_color='k', edge_linewidth=1, edge_color="#111", edge_alpha=.25, bgcolor="w",
            save=True, show=False, close=True,
            filepath=join(plot_dir, f'{row["id"]}_{row["city_ascii"]}_{row["iso2"]}.png')
        )


def process_row(row, output_dir, plot_dir, fallback_speed):
    res, g = construct_graph(row, fallback_speed)
    if res:
        store_graph(row, g, output_dir, plot_dir)
        logger.info('Stored graph for row {}', row)
    return row, res, g


def collect_existing_ids(output_dir):
    return {
        basename(filename).split('_')[0]
        for filename in glob.glob(join(output_dir, '*_*.graphml'))
    }


@logger.catch
def main(args):
    excluded_names = {x.lower() for x in args.exclude_cities}
    excluded_ids = collect_existing_ids(args.output_dir)
    countries = {x.lower() for x in args.country} if args.country is not None else None
    with open(args.worldcities_file.name, newline='') as cities_file:
        city_reader = csv.DictReader(cities_file)
        rows = [row for row in city_reader
                if row['city'].lower() not in excluded_names and row['id'] not in excluded_ids and (countries is None or row['iso2'].lower() in countries)]
    success, skipped, failures = set(), set(), list()
    logger.info('Submitting {} graph extraction jobs', len(rows))
    results = Parallel(n_jobs=args.max_workers, backend='multiprocessing')(
        delayed(process_row)(row, args.output_dir, args.plot_dir, args.fallback_speed) for row in rows
    )
    for row, is_success, output in results:
        if is_success is None:
            skipped.add(output)
            continue
        if is_success:
            success.add(row['id'])
        else:
            failures.append(output)
    logger.info('Scraping has been completed, {} graphs were generated, {} graphs were skipped', len(success), len(skipped))
    logger.info('Failed queries:\n{}', '\n'.join(f'- #{id}, {query}: {e}' for id, query, e in failures))


if __name__ == '__main__':
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.log_file = True
    parser = argparse.ArgumentParser(description='Collect a dataset of city road networks')
    parser.add_argument('--worldcities-file', '-i', type=argparse.FileType('r'), required=True,
                        help='Path to the WorldCities CSV file')
    parser.add_argument('--log-file', '-l', required=False, help='Write scraping log into the file')
    parser.add_argument('--fallback-speed', '-s', type=int, default=60, help='Default movement speed in km/h')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for GraphML files')
    parser.add_argument('--plot-dir', '-p', required=False, help='Output directory for city map plots')
    parser.add_argument('--exclude-cities', '-x', nargs='*', default=list(), help='List of cities excluded from dataset generation')
    parser.add_argument('--country', '-c', nargs='*', help='List of countries used for dataset generation')
    parser.add_argument('--max-workers', type=int, required=False, help='Maximum number of jobs run simultaneously')
    args = parser.parse_args()
    if args.log_file is not None:
        logger.add(args.log_file)
    main(args)
