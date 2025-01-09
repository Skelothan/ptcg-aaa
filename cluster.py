from __future__ import annotations

from config import CONFIG 
import copy
from datetime import datetime
import deck
from enum import Enum
import functools
import heapq
import itertools
import math
import multiprocessing as mp
import multiprocessing.queues as mpq
from typing import Iterable


class NestingSet:
    '''Set wrapper which keeps track of how many elements are in its great-great-etc. grandchildren.'''

    def __init__(self, s: Iterable, distance: float):
        self.set: set[deck.Deck | NestingSet] = set(s)

        self.size = 0
        self.contents: set[deck.Deck] = set()
        for i in s:
            if hasattr(i, "size") and hasattr(i, "contents"):
                self.size += i.size
                self.contents = self.contents.union(i.contents)
            else:
                self.size += 1
                self.contents.add(i)

        self.distance = distance
        self.stability: float | None = None
        self.deck_cluster: deck.DeckCluster
        self.cohesion: float

    def __hash__(self):
        return hash(tuple(sorted(self.contents)))
    
    def add(self, i):
        self.set.add(i)
        if hasattr(i, "size") and hasattr(i, "contents"):
            self.size += i.size
            self.contents = self.contents.union(i.contents)
        else:
            self.size += 1
            self.contents.add(i)

    def remove(self, i):
        self.set.remove(i)
        if hasattr(i, "size") and hasattr(i, "contents"):
            self.size -= i.size
            self.contents = self.contents.difference(i.contents)
        else:
            self.size -= 1
            self.contents.remove(i)

    def __repr__(self) -> str:
        if self.stability:
            return f"<NestingSet: size {self.size}, stability {round(self.stability, 2)}>"
        if self.deck_cluster and self.cohesion:
            return f"<NestingSet({self.deck_cluster.title}): size {self.size}, cohesion {round(self.cohesion, 2)}>"
        return f"<NestingSet: size {self.size}>"


class DisjointSetForest():

    def __init__(self, data: Iterable, linkages: list[tuple[float, deck.Deck, deck.Deck]]):
        '''
        linkages should be a heap.
        '''

        self.data = set(data)
        self.uf_forest = set(data)
        self.condensed_tree: dict[NestingSet, tuple[NestingSet, NestingSet]] = {}
        self.linkages = linkages

        self.selected_clusters = []
        self.rogue_decks = set(data)

    def union_find(self):
        
        total_size = len(self.data) - 1

        while len(self.data) > 1 and len(self.linkages) > 0:
            distance, d1, d2 = heapq.heappop(self.linkages)

            s1 = None
            s2 = None
            for s in self.data:
                if isinstance(s, NestingSet):
                    if d1 in s.contents:
                        s1 = s
                    if d2 in s.contents:
                        s2 = s
                else: # s is a Deck
                    if s == d1:
                        s1 = s
                    if s == d2:
                        s2 = s
                if s1 and s2:
                    break
                
            if s1 == s2: # Don't do anything if the two are already part of the same parent
                break
            else:
                merged_set = NestingSet((s1, s2), distance)
                self.data.remove(s1)
                self.data.remove(s2)
                self.data.add(merged_set)
            
            print(f"  Building cluster hierarchy... (Progress: {total_size - len(self.data) + 1}/{total_size})", end="\r")

        self.uf_forest = self.data.pop()
        print("")

    def _do_condense_tree(self, nesting_set: NestingSet, parent: NestingSet):
        print(f"  Condensing cluster hierarchy...", end="\r")

        def mark_death_distances(item: NestingSet | deck.Deck, distance: float):
            if isinstance(item, NestingSet):
                for d in item.contents:
                    d.death_distance = distance
            else:
                item.death_distance = distance

        item1, item2 = tuple(nesting_set.set)

        if all([
            isinstance(item1, NestingSet) and item1.size >= CONFIG["ROGUE_DECK_THRESHOLD"],
            isinstance(item2, NestingSet) and item2.size >= CONFIG["ROGUE_DECK_THRESHOLD"]
            ]):
            self.condensed_tree[parent] = (item1, item2)
            self._do_condense_tree(item1, item1)
            self._do_condense_tree(item2, item2)
        elif all([
            isinstance(item1, NestingSet) and item1.size >= CONFIG["ROGUE_DECK_THRESHOLD"],
            not(isinstance(item2, NestingSet) and item2.size >= CONFIG["ROGUE_DECK_THRESHOLD"])
        ]):
            mark_death_distances(item2, nesting_set.distance)
            self._do_condense_tree(item1, parent)
        elif all([
            not(isinstance(item1, NestingSet) and item1.size >= CONFIG["ROGUE_DECK_THRESHOLD"]),
            isinstance(item2, NestingSet) and item2.size >= CONFIG["ROGUE_DECK_THRESHOLD"]
        ]):
            mark_death_distances(item1, nesting_set.distance)
            self._do_condense_tree(item2, parent)
        else:
            mark_death_distances(item1, nesting_set.distance)
            mark_death_distances(item2, nesting_set.distance)

    def condense_tree(self):
        self._do_condense_tree(self.uf_forest, self.uf_forest)
        print("")

    def _do_select_clusters(self, considering: NestingSet, selected_clusters: set[NestingSet]):
        if considering in self.condensed_tree.keys():
            c1, c2 = self.condensed_tree[considering]
            self._do_select_clusters(c1, selected_clusters)
            self._do_select_clusters(c2, selected_clusters)

            if considering.stability >= c1.stability + c2.stability:
                selected_clusters.remove(c1)
                selected_clusters.remove(c2)
                selected_clusters.add(considering)
            else:
                considering.stability = c1.stability + c2.stability

    def select_clusters_stability(self, card_counter: deck.CardCounter):
        print(f"  Selecting clusters...", end="\r")

        all_sets = {x for y in self.condensed_tree.values() for x in y}
        all_sets.add(self.uf_forest)
       
        # Normally, you'd use stability to proceed here with HDBSCAN*. 
        # But I found that trying to group with stability pretty much always resulted in a single blobby archetype.
        # I've left my implementation here, but we'll be skipping it.
        warning_given = False
        for c in all_sets:
            if c.distance is None:
                c.stability = None
            total = 0
            for d in c.contents:
                if d.death_distance == 0:
                    if not warning_given:
                        print("  \033[93m[WARNING]\033[0m: A pair of decks had mutual reachability of 0. Your K-threshold is likely too low.")
                        warning_given = True
                    total += (float("inf") - 1 / c.distance)
                else:
                    total += (1 / d.death_distance - 1 / c.distance)
            c.stability = total

        selected_clusters = all_sets - self.condensed_tree.keys()

        self._do_select_clusters(self.uf_forest, selected_clusters)

        for c in selected_clusters:
            self.selected_clusters.append(c.deck_cluster)
            self.rogue_decks = self.rogue_decks.difference(c.deck_cluster.decks)

        print("")

    def select_clusters_cohesion(self, card_counter: deck.CardCounter):
        print(f"  Selecting clusters...", end="\r")

        all_sets = {x for y in self.condensed_tree.values() for x in y}
        all_sets.add(self.uf_forest)
        
        for c in all_sets:
            c.deck_cluster = functools.reduce(lambda x,y: x+y, c.contents)
            c.cohesion = sum([card_counter.get_deck_max_possible_inclusion_weighted_Jaccard(c.deck_cluster, d) for d in c.deck_cluster.decks]) / len(c.deck_cluster.decks)

        selected_clusters = all_sets - self.condensed_tree.keys()

        for c in selected_clusters:
            self.selected_clusters.append(c.deck_cluster)
            self.rogue_decks = self.rogue_decks.difference(c.deck_cluster.decks)

        print("")


class ClusterMethod(Enum):
    UPGMA = "UPGMA"
    HDBSCAN = "HDBSCAN*"


# TODO: refactor this — have each cluster algorithm use its own cluster engine subclass and initialize with a factory
class ClusterEngine:
    """
    Class that stores both decks and can cluster them together.

    Attributes
    ----------
    card_counter : deck.CardCounter
        A CardCounter containing card usage statistics of the decks in this ClusterEngine. Used to provide the similarity functions.
    decks : dict[str: deck.Deck]
        The original decks input into this `ClusterEngine`.
    decks_and_clusters : dict[str: Union[deck.Deck, deck.DeckCluster]]
        The final output of this `ClusterEngine`, containing decks and clusters. Initialized to a copy of decks; clusters are formed by calling a cluster method.
    similarities : dict[tuple[str, str]: float]
        Cache of deck pairs and their similarity (distance) values.
    greatest_similarity : float
        The similarity value of the two most similar decks in decks_and_clusters. Used during UPGMA clustering to determine which pair of decks/clusters to join next.
    most_similar_pair : tuple[str, str]
        Used during UPGMA clustering to determine which pair of decks/clusters to join next.
    """

    def __init__(self, card_counter: deck.CardCounter, decks: dict[str, deck.Deck]):
        self.card_counter = card_counter
        self.original_decks = decks
        self.decks_and_clusters: dict[str, deck.DeckLike] = copy.copy(decks)
        self.rogue_decks: set[deck.DeckLike] = set()
        self.similarities: dict[tuple[str, str], float] = {}

        self.greatest_similarity: float = -1
        self.most_similar_pair: tuple[str, str] = tuple()

        # self.mutual_reachabilities: list[float, deck.Deck, deck.Deck] = []
        self.spanning_tree_root: deck.Deck
        self.spanning_tree_distances: list[tuple[float, deck.Deck, deck.Deck]] = []

    def _update_most_similar_pair(self):
        """
        Updates the values of greatest_similarity and most_similar_pair.
        """
        self.most_similar_pair = max(self.similarities, key=lambda x: self.similarities[x])
        self.greatest_similarity = self.similarities[self.most_similar_pair]

    def _auto_cluster_identical_decks(self):
        """
        Takes the hash value of the contents all decks and creates clusters for any that are identical.
        This is generally faster than any of the other cluster algorithms and so improves efficiency.
        """
        print("Auto-clustering identical decks...")
        original_size = len(self.decks_and_clusters)
        deck_contents_cache: dict[int, deck.DeckLike] = {}
        for d in self.decks_and_clusters.values():
            if d.contents_hash() not in deck_contents_cache:
                deck_contents_cache[d.contents_hash()] = d
            else: # Combine the two
                deck_contents_cache[d.contents_hash()] = deck_contents_cache[d.contents_hash()] + d

        self.decks_and_clusters = {d.id: d for d in deck_contents_cache.values()}
        print(f"Identical decks clustered. (Reduced from {original_size} to {len(self.decks_and_clusters)} decks)")

    def _fill_queue(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], num_threads):
        """
        Producer process for initial similarity matrix build. Fills the tasks queue with pairs of deck/clusters.
        """
        similarities_to_calc = itertools.combinations(self.decks_and_clusters.values(), 2)
        num_tasks_queued = 0
        for pair in similarities_to_calc:
            tasks.put(pair)
            num_tasks_queued += 1
        stop_signals_queued = 0
        for signal in [None] * num_threads:
            tasks.put(signal)
            stop_signals_queued += 1

    def _compute_similarities(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], output: mpq.Queue[tuple[tuple[str, str], float]]):
        """
        Worker process for initial similarity matrix build. Calculates the similarity of the provided decks/clusters and put it in the output queue.
        """
        while True:
            pair = tasks.get(block=True)
            if pair is None:
                output.put(None)
                break
            d1, d2 = pair
            similarity = self.card_counter.get_deck_max_possible_inclusion_weighted_Jaccard(d1, d2) # TODO: make function choice configurable
            output.put(((min(d1.id, d2.id), max(d1.id, d2.id)), similarity))

    def _build_initial_similarity_matrix(self):
        """
        Fills `similarities` with the similarity of each pair of decks/clusters in the initial deck and cluster list.

        Multiprocessed to improve efficiency. The number of subprocesses is equal to CONFIG["NUM_THREADS"] plus one, plus the main process.
        """
        start_time = datetime.now()
        print("Building initial similarity matrix...")

        similarities_total_count = math.comb(len(self.decks_and_clusters), 2)
        
        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue()
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue()

        processes = []

        # Start task queue-filling producer process
        producer_process = mp.Process(target=self._fill_queue, args=(tasks, CONFIG.get("NUM_THREADS")))
        processes.append(producer_process)
        producer_process.start()

        # Start task queue-emptying worker processes
        for _ in range(CONFIG.get("NUM_THREADS")):
            process = mp.Process(target=self._compute_similarities, args=(tasks, outputs))
            processes.append(process)
            process.start()

        # Main process works on emptying output queue
        num_finished_processes = 0
        num_outputs_received = 0
        while True:
            output = outputs.get()
            if output is None:
                num_finished_processes += 1
                if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                    break
            else:
                num_outputs_received += 1
                self.similarities[output[0]] = output[1]
                d1 = self.decks_and_clusters[output[0][0]]
                d2 = self.decks_and_clusters[output[0][1]]
                heapq.heappush(d1.similarities, (output[1], d2))
                heapq.heappush(d2.similarities, (output[1], d1))
                print(f"  Calculated similarity for {output[0][0].ljust(32)} and {output[0][1].ljust(32)} (Progress: {num_outputs_received}/{similarities_total_count})", end="\r")

        end_time = datetime.now()
        print(f"\nSimilarity matrix built. (Time taken: {(end_time - start_time)})")

        self._update_most_similar_pair()

    def _do_upgma(self):
        """
        Main UPGMA logic.
        """
        start_time = datetime.now()
        print("Beginning clustering of decks with UPGMA method...")

        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue()
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue()

        merge_count = 0

        while self.greatest_similarity > CONFIG.get("CLUSTER_SIMILARITY_THRESHOLD"):
            # Merge the two most similar decks/clusters
            d1 = self.decks_and_clusters[self.most_similar_pair[0]]
            d2 = self.decks_and_clusters[self.most_similar_pair[1]]
            cluster = d1 + d2

            print(f"  Merging decks {d1.id.ljust(32)} and {d2.id.ljust(32)} (Similarity: {str(round(self.greatest_similarity, 4)).ljust(6, "0")}/{CONFIG.get("CLUSTER_SIMILARITY_THRESHOLD")}) ({merge_count} merged / {len(self.decks_and_clusters)} left)", end="\r")

            # Remove the old decks/clusters from the deck list
            del self.decks_and_clusters[d1.id]
            del self.decks_and_clusters[d2.id]

            # Update the similarities list
            for pair in list(self.similarities.keys()):
                if d1.id in pair or d2.id in pair:
                    del self.similarities[pair]

            # TODO: clean this mess up
            similarities_to_calc = [(cluster, dac_deck) for dac_deck in self.decks_and_clusters.values()]
            # If the number of similarities is too small, don't bother with the subprocesses
            if len(similarities_to_calc) > 10000 * CONFIG.get("NUM_THREADS"):

                similarities_to_calc = iter([(cluster, dac_deck) for dac_deck in self.decks_and_clusters.values()])

                processes = []
                for _ in range(CONFIG.get("NUM_THREADS")):
                    process = mp.Process(target=self._compute_similarities, args=(tasks, outputs))
                    processes.append(process)
                    process.start()

                for pair in similarities_to_calc:
                    tasks.put(pair)
                for signal in [None] * CONFIG.get("NUM_THREADS"):
                    tasks.put(signal)

                num_finished_processes = 0
                while True:
                    output = outputs.get()
                    if output is None:
                        num_finished_processes += 1
                        if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                            break
                    else:
                        self.similarities[output[0]] = output[1]
            else:
                for dac_id, dac_deck in self.decks_and_clusters.items():
                    self.similarities[(cluster.id, dac_id)] = self.card_counter.get_deck_max_possible_inclusion_weighted_Jaccard(cluster, dac_deck) # TODO: make the function configurable

            # Add the new cluster to the deck list
            self.decks_and_clusters[cluster.id] = cluster
            merge_count += 1

            self._update_most_similar_pair()

        end_time = datetime.now()
        print(f"\nFinished clustering. (Time taken: {(end_time - start_time)})")

    def cluster_upgma(self):
        """
        Cluster the decks in this `ClusterEngine`'s `decks_and_clusters` using UPGMA.
        Straightforward but very slow (O(n^2 logn)). Unreasonable on more than a few hundred decks.
        """
        # Auto-cluster any identical decks
        self._auto_cluster_identical_decks()

        # Initial similarity matrix build
        self._build_initial_similarity_matrix()

        # UPGMA
        self._do_upgma()

    def _mut_reach_fill_queue(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], num_threads):
        """
        Producer process for initial similarity matrix build. Fills the tasks queue with pairs of deck/clusters.
        """
        similarities_to_calc = self.similarities.items()
        num_tasks_queued = 0
        for pair in similarities_to_calc:
            tasks.put(pair)
            num_tasks_queued += 1
        stop_signals_queued = 0
        for signal in [None] * num_threads:
            tasks.put(signal)
            stop_signals_queued += 1

    def _compute_mut_reach(self, tasks: mpq.Queue[tuple[tuple[str, str], float] | None], output: mpq.Queue[tuple[tuple[str, str], float]]):
        """
        Worker process for initial similarity matrix build. Calculates the similarity of the provided decks/clusters and put it in the output queue.
        """
        while True:
            t = tasks.get(block=True)
            if t is None:
                output.put(None)
                break
            pair, similarity = t
            d1: deck.Deck = self.decks_and_clusters[pair[0]]
            d2: deck.Deck = self.decks_and_clusters[pair[1]]
            mut_reach = max([d1.k_distance(CONFIG["K_THRESHOLD"]), d2.k_distance(CONFIG["K_THRESHOLD"]), 0.5 - similarity])
            output.put(((min(d1.id, d2.id), max(d1.id, d2.id)), mut_reach))

    def _calculate_mutual_reachabilities(self):
        start_time = datetime.now()
        print("Calculating mutual reachabilities...")

        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue()
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue()

        processes = []

        # Start task queue-filling producer process
        producer_process = mp.Process(target=self._mut_reach_fill_queue, args=(tasks, CONFIG.get("NUM_THREADS")))
        processes.append(producer_process)
        producer_process.start()

        # Start task queue-emptying worker processes
        for _ in range(CONFIG.get("NUM_THREADS")):
            process = mp.Process(target=self._compute_mut_reach, args=(tasks, outputs))
            processes.append(process)
            process.start()

        num_finished_processes = 0
        num_outputs_received = 0
        while True:
            output = outputs.get()
            if output is None:
                num_finished_processes += 1
                if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                    break
            else:
                num_outputs_received += 1
                d1: deck.Deck = self.decks_and_clusters[output[0][0]]
                d2: deck.Deck = self.decks_and_clusters[output[0][1]]
                mut_reach = output[1]
                heapq.heappush(d1.mut_reach_similarities, (mut_reach, d1, d2))
                heapq.heappush(d2.mut_reach_similarities, (mut_reach, d2, d1))
                print(f"  Calculated mutual reachability for {d1.id} and {d2.id} (Progress: {num_outputs_received}/{len(self.similarities)})", end="\r")
            
        end_time = datetime.now()
        print(f"\nMutual reachabilities calculated. (Time taken: {(end_time - start_time)})")


    def _build_spanning_tree(self):
        '''
        Uses Prim's algorithm to build a spanning tree out of the mutual reachability values.
        '''
        start_time = datetime.now()
        print("Building spanning tree...")

        spanning_tree_decks: set[deck.Deck] = set()

        self.spanning_tree_root: deck.Deck = self.decks_and_clusters[next(iter(self.original_decks))]
        spanning_tree_decks.add(self.spanning_tree_root)
        self.tree_similarities = self.spanning_tree_root.mut_reach_similarities

        while len(spanning_tree_decks) < len(self.original_decks):
            mut_reach_dist, this_deck, other_deck = heapq.heappop(self.tree_similarities)
            # distance = 0.5 - self.similarities[min(this_deck.id, other_deck.id), max(this_deck.id, other_deck.id)]
            if other_deck not in spanning_tree_decks:
                spanning_tree_decks.add(other_deck)
                heapq.heappush(self.spanning_tree_distances, (mut_reach_dist, this_deck, other_deck))
                self.tree_similarities += other_deck.mut_reach_similarities
                heapq.heapify(self.tree_similarities)
                print(f"  Connected {other_deck.id} to the spanning tree (Progress: {len(spanning_tree_decks)}/{len(self.original_decks)})", end="\r")

        end_time = datetime.now()
        print(f"\nSpanning tree built. (Time taken: {(end_time - start_time)})")

    def _hdbscan_hierarchical_cluster(self):
        start_time = datetime.now()
        print("Beginning clustering of decks with HDBSCAN* method...")

        forest = DisjointSetForest(self.decks_and_clusters.values(), self.spanning_tree_distances)
        forest.union_find()

        forest.condense_tree()

        forest.select_clusters_cohesion(self.card_counter)

        end_time = datetime.now()
        print(f"\nFinished clustering. (Time taken: {(end_time - start_time)})")

        return forest

    def cluster_hdbscan(self):
        # Initial similarity matrix build
        self._build_initial_similarity_matrix()

        self._calculate_mutual_reachabilities()

        self._build_spanning_tree()

        forest = self._hdbscan_hierarchical_cluster()
        self.decks_and_clusters = {c.id: c for c in forest.selected_clusters}
        self.rogue_decks = forest.rogue_decks


    def cluster(self, cluster_method=ClusterMethod.HDBSCAN):
        """
        Cluster the decks in this `ClusterEngine`'s `decks_and_clusters` using the provided `ClusterMethod`.
        
        Parameters
        ----------
        cluster_method : ClusterMethod
            A clustering algorithm to use. Default is HDBSCAN*.
        """
        {
            ClusterMethod.UPGMA: self.cluster_upgma,
            ClusterMethod.HDBSCAN: self.cluster_hdbscan,
        }[cluster_method]()

    def rename_archetypes(self):
        for archetype in sorted(self.decks_and_clusters.values(), key=lambda a: a.num_decks, reverse=True):
            if not isinstance(archetype, deck.DeckCluster):
                continue

            longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))
            # longest_table_line_length = longest_card_name_length + len(" | Weight | Avg. count")

            print(f"{"Card Name".ljust(longest_card_name_length)} | {"Weight"} | {"Avg. count"}")
            print(f"{"-" * longest_card_name_length} | {"------"} | {"----------"}")
            archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage(archetype.decklist).items(), key=lambda p: p[1], reverse=True)

            i = 0
            for card, count in archetype_cards:
                print(f"{card.ljust(longest_card_name_length)} | {str(round(count, CONFIG.get("REPORT_DECIMAL_ROUNDING"))).ljust(4, "0").rjust(6)} | {str(round(archetype.decklist.get(card), CONFIG.get("REPORT_DECIMAL_ROUNDING"))).ljust(4, "0").rjust(10)}")
                i += 1
                if i >= 20:
                    break

            print("\nWhat should this archetype be called? (Enter nothing to skip.)")
            new_name = input("> ")
            if len(new_name.strip()) != 0:
                archetype.rename(new_name)
            print("")
        
        print("All archetypes renamed.")

    def print_cluster_report(self):
        """
        Prints a report of all clusters with size bigger than CONFIG["ROGUE_DECK_THRESHOLD"] and which characteristic cards they contain 
        to the reports directory.
        """
        archetype_count = 0
        filename = f"reports/{CONFIG.get("TOURNAMENT_FORMAT_FILTER")}_archetypes.txt"
        with open(filename, "w") as file:
            for archetype in sorted(self.decks_and_clusters.values(), key=lambda a: a.num_decks, reverse=True):
                if archetype.num_decks < CONFIG.get("ROGUE_DECK_THRESHOLD"):
                    continue

                archetype_count += 1
                longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))
                longest_table_line_length = longest_card_name_length + len(" | Weight | Avg. count")
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage(archetype.decklist).items(), key=lambda p: p[1], reverse=True)

                file.write(f"Archetype {archetype_count}: {archetype.title} ({archetype.num_decks} decks)\n")
                file.write("-" * longest_table_line_length + "\n")

                longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage(archetype.decklist).items(), key=lambda p: p[1], reverse=True)
                
                file.write(f"{"Card Name".ljust(longest_card_name_length)} | {"Weight"} | {"Avg. count"}\n")
                file.write(f"{"-" * longest_card_name_length} | {"------"} | {"----------"}\n")
                for card, count in archetype_cards:
                    file.write(f"{card.ljust(longest_card_name_length)} | {str(round(count, CONFIG.get("REPORT_DECIMAL_ROUNDING"))).ljust(4, "0").rjust(6)} | {str(round(archetype.decklist.get(card), CONFIG.get("REPORT_DECIMAL_ROUNDING"))).ljust(4, "0").rjust(10)}\n")
                
                file.write("-" * longest_table_line_length + "\n\n\n")

        print(f"Saved archetype report to {filename}.")