from typing import Tuple, List
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork


def all_lanelets_by_merging_predecessors_from_lanelet(lanelet: 'Lanelet',
                                                      network: 'LaneletNetwork', max_length: float = 150.0) \
        -> Tuple[List['Lanelet'], List[List[int]]]:
    """
    Computes all predecessor lanelets starting from a provided lanelet
    and merges them to a single lanelet for each route.

    :param lanelet: The lanelet to start from
    :param network: The network which contains all lanelets
    :param max_length: maximal length of merged lanelets can be provided
    :return: List of merged lanelets, Lists of lanelet ids of which each merged lanelet consists
    """
    assert isinstance(lanelet, Lanelet), '<Lanelet>: provided lanelet is not a valid Lanelet!'
    assert isinstance(network, LaneletNetwork), '<Lanelet>: provided lanelet network is not a ' \
                                                'valid lanelet network!'

    if lanelet.predecessor is None or len(lanelet.predecessor) == 0:
        return [lanelet], [[lanelet.lanelet_id]]

    merge_jobs = find_lanelet_predecessors_in_range(lanelet, network, max_length=max_length)
    merge_jobs = [[lanelet] + [network.find_lanelet_by_id(p) for p in path] for path in merge_jobs]

    # Create merged lanelets from paths
    merged_lanelets = []
    merge_jobs_final = []
    for path in merge_jobs:
        pred = path[0]
        merge_jobs_tmp = [pred.lanelet_id]
        for lanelet in path[1:]:
            merge_jobs_tmp.append(lanelet.lanelet_id)
            pred = Lanelet.merge_lanelets(pred, lanelet)

        merge_jobs_final.append(merge_jobs_tmp)
        merged_lanelets.append(pred)

    return merged_lanelets, merge_jobs_final


def find_lanelet_predecessors_in_range(lanelet: "Lanelet", lanelet_network: "LaneletNetwork",
                                       max_length=50.0) -> List[List[int]]:
    """
    Finds all possible predecessor paths (id sequences) within max_length.

    :param lanelet: lanelet
    :param lanelet_network: lanelet network
    :param max_length: abort once length of path is reached
    :return: list of lanelet IDs
    """
    paths = [[p] for p in lanelet.predecessor]
    paths_final = []
    lengths = [lanelet_network.find_lanelet_by_id(p).distance[-1] for p in lanelet.predecessor]
    while paths:
        paths_next = []
        lengths_next = []
        for p, le in zip(paths, lengths):
            predecessors = lanelet_network.find_lanelet_by_id(p[-1]).predecessor
            if not predecessors:
                paths_final.append(p)
            else:
                for pred in predecessors:
                    if pred in p or pred == lanelet.lanelet_id or le >= max_length:
                        # prevent loops and consider length of first predecessor
                        paths_final.append(p)
                        continue

                    l_next = le + lanelet_network.find_lanelet_by_id(pred).distance[-1]
                    if l_next < max_length:
                        paths_next.append(p + [pred])
                        lengths_next.append(l_next)
                    else:
                        paths_final.append(p + [pred])

        paths = paths_next
        lengths = lengths_next

    return paths_final
