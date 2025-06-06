from __future__ import annotations

import abc
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
import os
import pickle
import shelve
from typing import Iterable


class ClusterHierarchyNode:
    """Set wrapper which keeps track of how many elements are in its great-great-etc. grandchildren.

    It might be possible to rework this with 2-tuples, since the set here never has more than two elements.

    Attributes
    ----------
    set : set[deck.Deck | NestingSet]
        The set of data.
    size : int
        The number of leaf `Deck`s in this set.
    contents : set[deck.Deck]
        Flattened set containing all of the leaf `Decks` in this set and all its children.
    distance : float
        Mutual reachability distance that the items in the set are from each other.
    stability: float | None
        A measure of how "long-lived" the cluster represented by this `NestingSet` is — how quickly do child points leave this set?
        An option for cluster selection in the final step.
    self.deck_cluster: deck.DeckCluster
        A `DeckCluster` made up of this set's `contents`.
    cohesion: float
        The average similarity of each of this `deck_cluster`'s components to it. Doesn't really get used, but it's here just in case.
    """

    def __init__(self, s: tuple[deck.Deck | ClusterHierarchyNode, deck.Deck | ClusterHierarchyNode], distance: float):
        self._children = s

        self.size = 0
        self.contents: set[deck.Deck] = set() # This maybe could be a tuple
        for i in s:
            if hasattr(i, "size") and hasattr(i, "contents"):
                self.size += i.size
                self.contents = self.contents.union(i.contents)
            else:
                self.size += 1
                self.contents.add(i)

        self.distance = distance
        self.death_distance: float | None = None
        self.stability: float | None = None
        self.deck_cluster: deck.DeckCluster | None = None
        self.cohesion: float

        self._hash: int | None = None

    @property
    def children(self):
        return self._children
    
    def _get_hash(self):
        self._hash = hash(self.children)
        return self._hash

    def __hash__(self):
        return self._hash or self._get_hash()

    def __repr__(self) -> str:
        # if self.stability:
        #     return f"<ClusterHierarchyNode: size {self.size}, stability {round(self.stability, 2)}>"
        # if self.deck_cluster and self.cohesion:
        #     return f"<ClusterHierarchyNode({self.deck_cluster.title}): size {self.size}, cohesion {round(self.cohesion, 2)}>"
        if self.deck_cluster and self.death_distance:
            return f"<CH Node({self.deck_cluster.title}): size {self.size}, dist: {round(1 / self.distance, 2)}, death: {round(1 / self.death_distance, 2)}>"
        elif self.deck_cluster:
            return f"<CH Node({self.deck_cluster.title}): size {self.size}, dist: {round(1 / self.distance, 2)}>"
        return f"<CH Node: size {self.size}>"


class ClusterHierarchy():
    """ Data structure for representing the cluster hierarchy.
    
    Attributes
    ----------
    data : set[deck.Deck | NestingSet]
        The input set of decks. Modified destructively during cluster determination.
    linkages : list[tuple[float, deck.Deck, deck.Deck]]
        Input data of mutual reachability distances. Used to build the cluster hierarchy.
    root_node : NestingSet
        Root node of the hierarchical tree after `union_find()` is run.
    condensed_tree : dict[NestingSet, tuple[NestingSet, NestingSet]]
        Condensed hierarchical tree produced by `condense_tree()`. Produces larger clusters.
    selected_clusters : list[deck.DeckCluster]
        Our prize: the final output clusters.
    rogue_decks: set[deck.Deck]
        Group of decks that don't belong to any cluster.
    
    """

    def __init__(self, data: Iterable[deck.Deck], linkages: list[tuple[float, deck.Deck, deck.Deck]]):
        '''
        Linkages should be a heap.
        '''

        self.data: set[deck.Deck | ClusterHierarchyNode] = set(data)
        self.linkages = copy.copy(linkages)

        self.root_node: ClusterHierarchyNode
        self.condensed_tree: dict[ClusterHierarchyNode, tuple[ClusterHierarchyNode, ClusterHierarchyNode]] = {}
        
        self.selected_clusters: list[deck.DeckCluster] = []
        self.rogue_decks: set[deck.DeckLike] = set(data)
        self.rogue_clusters: set[deck.DeckCluster] = set()

    def build_hierarchy(self):
        """Produces a hierarchical cluster tree out of the linkage data."""
        total_size = len(self.data) - 1

        while len(self.data) > 1 and len(self.linkages) > 0:
            distance, d1, d2 = heapq.heappop(self.linkages)

            s1 = None
            s2 = None
            for s in self.data:
                if isinstance(s, ClusterHierarchyNode):
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
                continue
            else:
                merged_set = ClusterHierarchyNode((s1, s2), distance)
                self.data.remove(s1)
                self.data.remove(s2)
                self.data.add(merged_set)
            
            print(f"  Building cluster hierarchy... (Progress: {total_size - len(self.data) + 1}/{total_size})", end="\r")

        self.root_node: ClusterHierarchyNode = self.data.pop()
        print("")

    def _mark_death_distances(self, item: ClusterHierarchyNode | deck.Deck, distance: float):
        if isinstance(item, ClusterHierarchyNode):
            item.death_distance = distance
            for d in item.contents:
                d.death_distance = distance
        else:
            item.death_distance = distance

    def _do_condense_tree_recursive(self, current_node: ClusterHierarchyNode, parent: ClusterHierarchyNode):
        child1, child2 = tuple(current_node.children)

        if all([
            isinstance(child1, ClusterHierarchyNode) and child1.size >= CONFIG["ROGUE_DECK_THRESHOLD"],
            isinstance(child2, ClusterHierarchyNode) and child2.size >= CONFIG["ROGUE_DECK_THRESHOLD"]
            ]):
            self.condensed_tree[parent] = (child1, child2)
            self._do_condense_tree_recursive(child1, child1)
            self._do_condense_tree_recursive(child2, child2)
        elif all([
            isinstance(child1, ClusterHierarchyNode) and child1.size >= CONFIG["ROGUE_DECK_THRESHOLD"],
            not(isinstance(child2, ClusterHierarchyNode) and child2.size >= CONFIG["ROGUE_DECK_THRESHOLD"])
        ]):
            self._mark_death_distances(child2, current_node.distance)
            self._do_condense_tree_recursive(child1, parent)
        elif all([
            not(isinstance(child1, ClusterHierarchyNode) and child1.size >= CONFIG["ROGUE_DECK_THRESHOLD"]),
            isinstance(child2, ClusterHierarchyNode) and child2.size >= CONFIG["ROGUE_DECK_THRESHOLD"]
        ]):
            self._mark_death_distances(child1, current_node.distance)
            self._do_condense_tree_recursive(child2, parent)
        else:
            self._mark_death_distances(child1, current_node.distance)
            self._mark_death_distances(child2, current_node.distance)

    def _do_condense_tree_iterative(self):
        tasks: list[tuple[ClusterHierarchyNode, ClusterHierarchyNode]] = []
        tasks.append((self.root_node, self.root_node))
        self.rogue_decks = set()

        while len(tasks) > 0:
            current_node, parent = tasks.pop()
            child1, child2 = tuple(current_node.children)

            child_sizes = 2 * (isinstance(child1, ClusterHierarchyNode) and child1.size >= CONFIG["ROGUE_DECK_THRESHOLD"]) + \
                (isinstance(child2, ClusterHierarchyNode) and child2.size >= CONFIG["ROGUE_DECK_THRESHOLD"])
            
            match child_sizes:
                case 3: # True split
                    self.condensed_tree[parent] = (child1, child2)
                    parent.death_distance = current_node.distance
                    tasks.append((child2, child2))
                    tasks.append((child1, child1))
                case 2:
                    self._mark_death_distances(child2, current_node.distance)
                    tasks.append((child1, parent))
                    
                    if isinstance(child2, ClusterHierarchyNode):
                        child2.deck_cluster = functools.reduce(lambda x,y: x+y, child2.contents)
                        self.rogue_clusters.add(child2.deck_cluster)
                    else:
                        self.rogue_decks.add(child2)
                case 1:
                    self._mark_death_distances(child1, current_node.distance)
                    tasks.append((child2, parent))
                    
                    if isinstance(child1, ClusterHierarchyNode):
                        child1.deck_cluster = functools.reduce(lambda x,y: x+y, child1.contents)
                        self.rogue_clusters.add(child1.deck_cluster)
                    else:
                        self.rogue_decks.add(child1)
                case 0:
                    self._mark_death_distances(child1, current_node.distance)
                    self._mark_death_distances(child2, current_node.distance)
                    
                    if isinstance(child1, ClusterHierarchyNode):
                        child1.deck_cluster = functools.reduce(lambda x,y: x+y, child1.contents)
                        self.rogue_clusters.add(child1.deck_cluster)
                    else:
                        self.rogue_decks.add(child1)
                    
                    if isinstance(child2, ClusterHierarchyNode):
                        child2.deck_cluster = functools.reduce(lambda x,y: x+y, child2.contents)
                        self.rogue_clusters.add(child2.deck_cluster)
                    else:
                        self.rogue_decks.add(child2)
                case _:
                    raise ValueError(f"Got an unusual child size case when looking at node {current_node} (parent {parent})")

    def condense_tree(self):
        """Condenses the cluster hierarchy into large blobs by ignoring small groupings."""
        print(f"  Condensing cluster hierarchy...", end="\r")
        self._do_condense_tree_iterative()
        print("")

    def _do_select_clusters(self, considering: ClusterHierarchyNode, selected_clusters: set[ClusterHierarchyNode]):
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

    def select_clusters_stability(self):
        """Select clusters based on their stability; i.e. how often decks fall out of the cluster as distance increases.
        
        This would be the more traditional HDBSCAN* cluster selection method. I found it didn't work well on my smaller datasets, so I'm electing not to use it.
        """
        print(f"  Selecting clusters...", end="\r")

        all_sets = {x for y in self.condensed_tree.values() for x in y}
        all_sets.add(self.root_node)
       
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

        self._do_select_clusters(self.root_node, selected_clusters)

        for c in selected_clusters:
            self.selected_clusters.append(c.deck_cluster)
            # self.rogue_decks = self.rogue_decks.difference(c.deck_cluster.decks)

        print("")

    def select_clusters_cohesion(self, card_counter: deck.CardCounter):
        """Select clusters based on cohesion, or how similar on average the decks in the cluster are to the collective.

        Or it would, except cohesion always seems to go down pretty drastically, so I've currently got it to just pick the smallest identified groupings. 
        Might need to change that later — we'll see how different datasets goes.
        
        Parameters
        ----------
        card_counter : deck.CardCounter
            A CardCounter of the average card counts in the entire deck population. Needed to weight decks for similarity calculations. 
        """
        print(f"  Selecting clusters...", end="\r")

        all_sets = {x for y in self.condensed_tree.values() for x in y}
        all_sets.add(self.root_node)
        
        # TODO: make the weighting function configurable
        # for c in all_sets:
        #     c.deck_cluster = functools.reduce(lambda x,y: x+y, c.contents)
        #     c.cohesion = sum([card_counter.get_deck_max_possible_inclusion_adjusted_weighted_Jaccard(c.deck_cluster, d) for d in c.deck_cluster.decks]) / len(c.deck_cluster.decks)

        selected_clusters = all_sets - self.condensed_tree.keys()

        for c in all_sets:
            c.deck_cluster = functools.reduce(lambda x,y: x+y, c.contents)

        for c in selected_clusters:
            c.deck_cluster = functools.reduce(lambda x,y: x+y, c.contents)
            self.selected_clusters.append(c.deck_cluster)
            self.rogue_decks = self.rogue_decks.difference(c.deck_cluster.decks)
            self.rogue_clusters = {rc for rc in self.rogue_clusters if len(rc.decks.intersection(c.deck_cluster.decks)) == 0}

        self.rogue_decks = self.rogue_decks.union(self.rogue_clusters)

        print("")


class ClusterMethod(Enum):
    UPGMA = "UPGMA"
    HDBSCAN = "HDBSCAN*"


class ClusterEngine(metaclass=abc.ABCMeta):
    """
    Class that stores both decks and can cluster them together.

    Attributes
    ----------
    card_counter : deck.CardCounter
        A CardCounter containing card usage statistics of the decks in this ClusterEngine. Used to provide the similarity functions.
    original_decks : dict[str: deck.Deck]
        The original decks input into this `ClusterEngine`.
    decks_and_clusters: dict[str: deck.DeckLike]
        Working directory used during calculation.
    clusters: dict[str: deck.DeckCluster]
        The final output of this `ClusterEngine`, containing decks and clusters. Initialized to a copy of decks; clusters are formed by calling a cluster method.
    rogue_decks: set[deck.Deck]
        The other output of this `ClusterEngine` — decks that did not fit any archetype.
    similarities : dict[tuple[str, str]: float]
        Cache of deck pairs and their similarity (distance) values.
    """

    def __init__(self, card_counter: deck.CardCounter, decks: dict[str, deck.Deck]):
        self.card_counter = card_counter
        self.original_decks = decks
        self.decks_and_clusters: dict[str, deck.DeckLike] = copy.copy(decks)
        self.decks_and_clusters_by_contents: dict[str, deck.DeckLike]
        self.clusters: dict[str, deck.DeckCluster]
        self.rogue_decks: set[deck.DeckLike] = set()
        self.similarities: dict[tuple[str, str], float] = {}

    def _auto_cluster_identical_decks(self):
        """
        Takes the hash value of the contents all decks and creates clusters for any that are identical.
        This is generally faster than any of the other cluster algorithms and so improves efficiency.
        Additionally used to suggest a minimum K-threshold for HDBSCAN* clustering.
        """
        print("Auto-clustering identical decks...")
        original_size = len(self.decks_and_clusters)
        deck_contents_cache: dict[int, deck.DeckLike] = {}
        for d in self.decks_and_clusters.values():
            if d.contents_hash not in deck_contents_cache:
                deck_contents_cache[d.contents_hash] = d
            else: # Combine the two
                deck_contents_cache[d.contents_hash] = deck_contents_cache[d.contents_hash] + d

        self.decks_and_clusters = {d.id: d for d in deck_contents_cache.values()}
        self.decks_and_clusters_by_contents = deck_contents_cache

        print(f"Identical decks clustered. (Reduced from {original_size} to {len(self.decks_and_clusters)} decks)")

    def _fill_queue(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], num_threads):
        """
        Producer process for initial similarity matrix build. Fills the tasks queue with pairs of deck/clusters.
        """
        similarities_to_calc = itertools.combinations(self.decks_and_clusters.values(), 2)
        num_tasks_queued = 0
        for pair in similarities_to_calc:
            tasks.put(pair, block=True)
            num_tasks_queued += 1
        stop_signals_queued = 0
        for signal in [None] * num_threads:
            tasks.put(signal, block=True)
            stop_signals_queued += 1

    def _compute_similarities(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], output: mpq.Queue[tuple[tuple[str, str], float]]):
        """
        Worker process for initial similarity matrix build. Calculates the similarity of the provided decks/clusters and put it in the output queue.
        """
        while True:
            pair = tasks.get(block=True)
            if pair is None:
                output.put(None, block=True)
                break
            d1, d2 = pair
            similarity = self.card_counter.get_deck_max_possible_inclusion_adjusted_weighted_Jaccard(d1, d2) # TODO: make function choice configurable
            output.put(((min(d1.id, d2.id), max(d1.id, d2.id)), similarity), block=True)

    def _build_initial_similarity_matrix(self):
        """
        Fills `similarities` with the similarity of each pair of decks/clusters in the initial deck and cluster list.

        Multiprocessed to improve efficiency. The number of subprocesses is equal to CONFIG["NUM_THREADS"] plus one, plus the main process.
        """
        start_time = datetime.now()
        print("Building initial similarity matrix...")

        similarities_total_count = math.comb(len(self.decks_and_clusters), 2)
        
        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue(maxsize=100000)
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue(maxsize=100000)

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
            output = outputs.get(block=True)
            if output is None:
                num_finished_processes += 1
                if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                    break
            else:
                num_outputs_received += 1
                self.similarities[output[0]] = output[1]
                print(f"  Calculated similarity for {output[0][0].ljust(32)} and {output[0][1].ljust(32)} (Progress: {num_outputs_received}/{similarities_total_count})", end="\r")

        end_time = datetime.now()
        print(f"\nSimilarity matrix built. (Time taken: {(end_time - start_time)})")

    def rename_archetypes(self):
        for archetype in sorted(self.clusters.values(), key=lambda a: a.num_decks, reverse=True):
            longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))

            print(f"{'Card Name'.ljust(longest_card_name_length)} | {'Weight'} | {'Avg. count'}")
            print(f"{'-' * longest_card_name_length} | {'------'} | {'----------'}")
            archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage_adjusted(archetype.decklist).items(), key=lambda p: p[1], reverse=True)

            i = 0
            for card, count in archetype_cards:
                print(f"{card.ljust(longest_card_name_length)} | {str(round(count, CONFIG.get('REPORT_DECIMAL_ROUNDING'))).ljust(4, '0').rjust(6)} | {str(round(archetype.decklist.get(card), CONFIG.get('REPORT_DECIMAL_ROUNDING'))).ljust(4, '0').rjust(10)}")
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
        filename = f"reports/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_archetypes.txt"
        with open(filename, "w") as file:
            for archetype in sorted(self.clusters.values(), key=lambda a: a.num_decks, reverse=True):
                if archetype.num_decks < CONFIG.get("ROGUE_DECK_THRESHOLD"):
                    continue

                archetype_count += 1
                longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))
                longest_table_line_length = longest_card_name_length + len(" | Weight | Avg. count")
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage_adjusted(archetype.decklist).items(), key=lambda p: p[1], reverse=True)

                file.write(f"Archetype {archetype_count}: {archetype.title} ({archetype.num_decks} decks | {round(archetype.num_decks / len(self.original_decks) * 100, 1)}%)\n")
                file.write("-" * longest_table_line_length + "\n")

                longest_card_name_length = max(len(max(archetype.decklist.keys(), key=len)), len("Card Name"))
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage_adjusted(archetype.decklist).items(), key=lambda p: p[1], reverse=True)
                
                file.write(f"{'Card Name'.ljust(longest_card_name_length)} | {'Weight'} | {'Avg. count'}\n")
                file.write(f"{'-' * longest_card_name_length} | {'------'} | {'----------'}\n")
                for card, count in archetype_cards:
                    file.write(f"{card.ljust(longest_card_name_length)} | {str(round(count, CONFIG.get('REPORT_DECIMAL_ROUNDING'))).ljust(4, '0').rjust(6)} | {str(round(archetype.decklist.get(card), CONFIG.get('REPORT_DECIMAL_ROUNDING'))).ljust(4, '0').rjust(10)}\n")
                
                file.write("-" * longest_table_line_length + "\n\n\n")

        print(f"Saved archetype report to {filename}.")

    def _get_num_rogue_decks(self):
        return functools.reduce(lambda x,y:x+y, map(lambda z: z.num_decks, self.rogue_decks))

    def print_rogue_deck_report(self):
        """
        Prints a list of all rogue decks.
        """
        filename = f"reports/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_rogue_decks.txt"
        with open(filename, "w") as file:
            file.write(f"Rogue Deck Report: {self._get_num_rogue_decks()} rogue archetypes\n\n")
            for decklike in sorted(self.rogue_decks, key=lambda a: a.num_decks, reverse=True):
                longest_card_name_length = max(len(max(decklike.decklist.keys(), key=len)), len("Card Name"))
                longest_table_line_length = longest_card_name_length + len(" | Weight | Count")
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage_adjusted(decklike.decklist).items(), key=lambda p: p[1], reverse=True)

                if isinstance(decklike, deck.DeckCluster):
                    file.write(f"{decklike.title} ({decklike.num_decks} decks | {round(decklike.num_decks / len(self.original_decks) * 100, 1)}%)\n")
                else:
                    file.write(f"{decklike.title}\n")
                file.write("-" * longest_table_line_length + "\n")

                longest_card_name_length = max(len(max(decklike.decklist.keys(), key=len)), len("Card Name"))
                archetype_cards = sorted(self.card_counter.weight_cards_by_max_possible_usage_adjusted(decklike.decklist).items(), key=lambda p: p[1], reverse=True)
                
                file.write(f"{'Card Name'.ljust(longest_card_name_length)} | {'Weight'} | {'Count'}\n")
                file.write(f"{'-' * longest_card_name_length} | {'------'} | {'-----'}\n")
                for card, count in archetype_cards:
                    file.write(f"{card.ljust(longest_card_name_length)} | {str(round(count, CONFIG.get('REPORT_DECIMAL_ROUNDING'))).ljust(4, '0').rjust(6)} | {str(round(decklike.decklist.get(card), CONFIG.get('REPORT_DECIMAL_ROUNDING'))).rjust(5)}\n")
                
                file.write("-" * longest_table_line_length + "\n\n\n")

        print(f"Saved rogue deck report to {filename}.")
    
    def print_metagame_report(self):
        archetype_count = 0
        filename = f"reports/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_metagame.txt"
        with open(filename, "w") as file:
            file.write("Archetype meta share:\n")

            longest_archetype_name_length = len(max(self.clusters.values(), key=lambda a: len(a.title)).title)
            longest_num_decks_length = len(str(max(self.clusters.values(), key=lambda a: a.num_decks).num_decks))
            archetype_count_length = len(str(len(self.clusters.values())))

            for archetype in sorted(self.clusters.values(), key=lambda a: a.num_decks, reverse=True):
                if archetype.num_decks < CONFIG.get("ROGUE_DECK_THRESHOLD"):
                    continue
                archetype_count += 1
                file.write(f"Archetype {str(archetype_count).rjust(archetype_count_length)}: {archetype.title.ljust(longest_archetype_name_length)} ({str(archetype.num_decks).rjust(longest_num_decks_length)} decks | {str(round(archetype.num_decks / len(self.original_decks) * 100, 1)).rjust(4)}%)\n")

            file.write(f"\n\nRogue deck percentage: {round(self._get_num_rogue_decks() / len(self.original_decks) * 100, 1)} %\n")

            min_majority_count = 0
            min_majority_share = 0
            for archetype in sorted(self.clusters.values(), key=lambda a: a.num_decks, reverse=True):
                min_majority_share += archetype.num_decks / len(self.original_decks)
                min_majority_count += 1
                if min_majority_share >= 0.5:
                    break
            file.write(f"\nMinimum majority count: {min_majority_count} \n")

        print(f"Saved metagame report to {filename}.")

    @abc.abstractmethod
    def cluster():
        raise NotImplementedError


class UPGMAClusterEngine(ClusterEngine):
    """
    `ClusterEngine` that can do UPGMA clustering. This is slow, but simple and generally effective on small datasets.

    Attributes
    ----------
    card_counter : deck.CardCounter
        A CardCounter containing card usage statistics of the decks in this ClusterEngine. Used to provide the similarity functions.
    original_decks : dict[str: deck.Deck]
        The original decks input into this `ClusterEngine`.
    clusters: dict[str: deck.DeckCluster]
        The final output of this `ClusterEngine`, containing decks and clusters. Initialized to a copy of decks; clusters are formed by calling a cluster method.
    rogue_decks: set[deck.Deck]
        The other output of this `ClusterEngine` — decks that did not fit any archetype.
    similarities : dict[tuple[str, str]: float]
        Cache of deck pairs and their similarity (distance) values.
    greatest_similarity : float
        The similarity value of the two most similar decks in decks_and_clusters. Used during UPGMA clustering to determine which pair of decks/clusters to join next.
    most_similar_pair : tuple[str, str]
        Used during UPGMA clustering to determine which pair of decks/clusters to join next.
    """
    def __init__(self, card_counter: deck.CardCounter, decks: dict[str, deck.Deck]):
        super().__init__(card_counter, decks)

        self.greatest_similarity: float = -1
        self.most_similar_pair: tuple[str, str] = tuple()

    def _update_most_similar_pair(self):
        """
        Updates the values of greatest_similarity and most_similar_pair.
        """
        self.most_similar_pair = max(self.similarities, key=lambda x: self.similarities[x])
        self.greatest_similarity = self.similarities[self.most_similar_pair]

    def _do_upgma(self):
        """
        Main UPGMA logic.
        """
        start_time = datetime.now()
        print("Beginning clustering of decks with UPGMA method...")

        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue(maxsize=10000)
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue(maxsize=10000)

        merge_count = 0

        while self.greatest_similarity > CONFIG.get("CLUSTER_SIMILARITY_THRESHOLD"):
            # Merge the two most similar decks/clusters
            d1 = self.decks_and_clusters[self.most_similar_pair[0]]
            d2 = self.decks_and_clusters[self.most_similar_pair[1]]
            cluster = d1 + d2

            print(f"  Merging decks {d1.id.ljust(32)} and {d2.id.ljust(32)} (Similarity: {str(round(self.greatest_similarity, 4)).ljust(6, '0')}/{CONFIG.get('CLUSTER_SIMILARITY_THRESHOLD')}) ({merge_count} merged / {len(self.decks_and_clusters)} left)", end="\r")

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
                    tasks.put(pair, block=True)
                for signal in [None] * CONFIG.get("NUM_THREADS"):
                    tasks.put(signal, block=True)

                num_finished_processes = 0
                while True:
                    output = outputs.get(block=True)
                    if output is None:
                        num_finished_processes += 1
                        if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                            break
                    else:
                        self.similarities[output[0]] = output[1]
            else:
                for dac_id, dac_deck in self.decks_and_clusters.items():
                    self.similarities[(cluster.id, dac_id)] = self.card_counter.get_deck_max_possible_inclusion_adjusted_weighted_Jaccard(cluster, dac_deck) # TODO: make the function configurable

            # Add the new cluster to the deck list
            self.decks_and_clusters[cluster.id] = cluster
            merge_count += 1

            self._update_most_similar_pair()

        end_time = datetime.now()
        print(f"\nFinished clustering. (Time taken: {(end_time - start_time)})")

    def cluster(self):
        """
        Cluster the decks in this `ClusterEngine`'s `decks_and_clusters` using UPGMA.
        Straightforward but very slow (O(n^2 logn)). Unreasonable on more than a few hundred decks.
        """
        # Auto-cluster any identical decks
        self._auto_cluster_identical_decks()

        # Initial similarity matrix build
        self._build_initial_similarity_matrix()
        self._update_most_similar_pair()

        # UPGMA
        self._do_upgma()

        for id, dl in self.decks_and_clusters.items():
            if isinstance(dl, deck.Deck):
                self.rogue_decks.add(dl)
            else: # dl is a DeckCluster
                self.clusters[id] = dl


class HDBSCANClusterEngine(ClusterEngine):
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
    spanning_tree_root : deck.Deck
        The root node of the minimum spanning tree constructed using mutual reachability.
    spanning_tree_distances : list[tuple[float, deck.Deck, deck.Deck]]
        Heapqueue of mutual reachability distances and deck pairs.
    """

    def __init__(self, card_counter: deck.CardCounter, decks: dict[str, deck.Deck]):
        super().__init__(card_counter, decks)

        self.spanning_tree_root: deck.Deck
        self.spanning_tree: list[tuple[float, deck.Deck, deck.Deck]] = []
        self.cluster_hierarchy: ClusterHierarchy | None = None
        
        self.k_similarities: dict[str, list[tuple[float, str]]] = {}

        self.similarities_calculated = False
        self.spanning_tree_built = False
        self.clusters_calculated = False

        self.SAVE_PATH = f"saved_data/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_K-{CONFIG.get('K_THRESHOLD')}"
        self.similarity_shelf_path: str = f"{self.SAVE_PATH}.aaasimshelf"
        self.mut_reach_shelf_path: str = f"{self.SAVE_PATH}.aaamrshelf"

    def _compute_similarities(self, tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None], output: mpq.Queue[tuple[tuple[str, str], float]]):
        """
        Worker process for initial similarity matrix build. Calculates the similarity of the provided decks/clusters and put it in the output queue.
        """
        while True:
            pair = tasks.get(block=True)
            if pair is None:
                output.put(None, block=True)
                break
            d1, d2 = pair

            similarity = self.card_counter.get_deck_max_possible_inclusion_adjusted_weighted_Jaccard(d1, d2) # TODO: make function choice configurable

            output.put(((min(d1.contents_hash, d2.contents_hash), max(d1.contents_hash, d2.contents_hash)), similarity), block=True)

    def load_similarities(self):
        filename = self.SAVE_PATH + ".aaasim"
        print(f"Loading similarities from {filename}...")
        with open(filename, "rb") as file:
            pickler = pickle.Unpickler(file)
            (self.decks_and_clusters_by_contents, self.similarities) = pickler.load()

    def load_spanning_tree(self):
        filename = self.SAVE_PATH + ".aaatree"
        print(f"Loading spanning tree from {filename}...")
        with open(filename, "rb") as file:
            pickler = pickle.Unpickler(file)
            (self.spanning_tree_root, self.spanning_tree) = pickler.load()

    def load_cluster_hierarchy(self):
        filename = self.SAVE_PATH + ".aaaclusters"
        print(f"Loading cluster hierarchy from {filename}...")
        with open(filename, "rb") as file:
            pickler = pickle.Unpickler(file)
            (self.clusters, self.rogue_decks) = pickler.load()

    def k_similarity_push(self, contents_hash, similarity: tuple[float, str]):
        """
        Writes a similarity of the deck/cluster with provided content hash to a shelf of similarities specifically for K-distance calculation.

        Will only store up to CONFIG["K_THRESHOLD"] values — others are not needed. Extra values will be deleted to reduce storage usage.
        
        Parameters
        ----------
        similarity : tuple[float, str]
            The similarity (float) of this Deck to the Deck with content hash (str).
        """
        if contents_hash not in self.k_similarities.keys():
            self.k_similarities[contents_hash] = []

        this_decklike: deck.DeckLike = self.decks_and_clusters_by_contents[contents_hash]
        other_decklike: deck.DeckLike = self.decks_and_clusters_by_contents[similarity[1]]
            
        for _ in range(other_decklike.num_decks):
            self.k_similarities[contents_hash].append(similarity)
        self.k_similarities[contents_hash].sort(reverse=True)
        del self.k_similarities[contents_hash][(CONFIG["K_THRESHOLD"] - this_decklike.num_decks + 1):]

    def _build_initial_similarity_matrix(self):
        """
        Fills `similarities` with the similarity of each pair of decks/clusters in the initial deck and cluster list.

        Multiprocessed to improve efficiency. The number of subprocesses is equal to CONFIG["NUM_THREADS"] plus one, plus the main process.
        """
        start_time = datetime.now()
        print("Building initial similarity matrix...")

        if os.path.isfile(self.similarity_shelf_path):
            os.remove(self.similarity_shelf_path)
        # if os.path.isfile(self.k_similarity_shelf_path):
        #     os.remove(self.k_similarity_shelf_path)

        similarities_total_count = math.comb(len(self.decks_and_clusters_by_contents), 2)

        manager = mp.Manager()
        tasks: mpq.Queue[tuple[deck.DeckLike, deck.DeckLike] | None] = manager.Queue(maxsize=10000)
        outputs: mpq.Queue[tuple[tuple[str, str], float]] = manager.Queue(maxsize=10000)

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
            with shelve.open(self.similarity_shelf_path, flag="c") as similarity_shelf:
                output = outputs.get(block=True)
                if output is None:
                    num_finished_processes += 1
                    if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                        break
                else:
                    num_outputs_received += 1
                    similarity_shelf[f"{output[0][0]},{output[0][1]}"] = output[1]

                    d1: deck.DeckLike = self.decks_and_clusters_by_contents[output[0][0]]
                    d2: deck.DeckLike = self.decks_and_clusters_by_contents[output[0][1]]

                    self.k_similarity_push(d1.contents_hash, (output[1], d2.contents_hash))
                    self.k_similarity_push(d2.contents_hash, (output[1], d1.contents_hash))

                    print(f"  Calculated similarity for {d1.id.ljust(32)} and {d2.id.ljust(32)} (Progress: {num_outputs_received}/{similarities_total_count})", end="\r")

        print("")
        count = 0
        for content_hash, d in self.decks_and_clusters_by_contents.items():
            d._k_distance = 0.5 - self.k_similarities[content_hash][-1][0]
            count += 1
            del self.k_similarities[content_hash]
            print(f"  Cached k-distance for {d.id.ljust(32)}... (Progress: {count}/{len(self.decks_and_clusters_by_contents)})", end="\r")


        end_time = datetime.now()
        print(f"\nSimilarity matrix built. (Time taken: {(end_time - start_time)})")

        filename = self.SAVE_PATH + ".aaasim"
        print(f"Saving similarities to {filename}...")
        with open(filename, "wb") as file:
            pickler = pickle.Pickler(file)
            pickler.dump((self.decks_and_clusters_by_contents, self.similarities))

    def _mut_reach_fill_queue(self, tasks: mpq.Queue[tuple[str, str] | None], num_threads):
        """
        Producer process for mutual reachability calculation. Fills the tasks queue with pairs of decks.
        """
        similarities_to_calc = itertools.combinations(self.decks_and_clusters_by_contents.keys(), 2)

        num_tasks_queued = 0
        for pair in similarities_to_calc:
            pair = (min(pair[0], pair[1]), max([pair[0], pair[1]]))
            tasks.put((pair), block=True)
            num_tasks_queued += 1
        stop_signals_queued = 0
        for signal in [None] * num_threads:
            tasks.put(signal, block=True)
            stop_signals_queued += 1

    def _compute_mut_reach(self, tasks: mpq.Queue[tuple[str, str] | None], output: mpq.Queue[tuple[tuple[str, str], float, tuple[str, str]]]):
        """
        Worker process for mutual reachability calculation. Calculates the similarity of the provided decks/clusters and put it in the output queue.
        """
        while True:
            with shelve.open(self.similarity_shelf_path, flag="r") as similarity_shelf:
                t = tasks.get(block=True)
                if t is None:
                    output.put(None, block=True)
                    break
                pair = t
                similarity = similarity_shelf[f"{pair[0]},{pair[1]}"]
                d1 = self.decks_and_clusters_by_contents[pair[0]]
                d2 = self.decks_and_clusters_by_contents[pair[1]]
                mut_reach = max(d1.k_distance, d2.k_distance, 0.5 - similarity)

                output.put((pair, mut_reach, (d1.id, d2.id)), block=True)

    def _calculate_all_mutual_reachabilities(self):
        """
        Calculates mutual reachability for each pair of decks from the previously calculated distances.
        This is what lets this algorithm find clusters of varying densities.
        It also essentially boots outlier points (rogue decks) further away from other data so it can be more easily ignored,
        and adjusts for identical decks (if your K-threshold is large enough).

        Mutual reachability values are stored on the `Deck`s themselves; this makes it easier to build a graph with them.

        Multiprocessed to improve efficiency. The number of subprocesses is equal to CONFIG["NUM_THREADS"] plus one, plus the main process.
        """
        start_time = datetime.now()
        print("Calculating mutual reachabilities...")

        if os.path.isfile(self.mut_reach_shelf_path):
            os.remove(self.mut_reach_shelf_path)

        similarities_total_count = math.comb(len(self.decks_and_clusters_by_contents), 2)

        manager = mp.Manager()
        tasks: mpq.Queue[tuple[str, str] | None] = manager.Queue(maxsize=10000)
        outputs: mpq.Queue[tuple[tuple[str, str], float, tuple[str, str]]] = manager.Queue(maxsize=10000)

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
            with shelve.open(self.mut_reach_shelf_path, flag="c") as mut_reach_shelf:
                output = outputs.get(block=True)
                if output is None:
                    num_finished_processes += 1
                    if num_finished_processes >= CONFIG.get("NUM_THREADS"):
                        break
                else:
                    num_outputs_received += 1

                    d1_content_hash, d2_content_hash = output[0]
                    mut_reach = output[1]

                    d1_mut_reach: list[float, str, str] = mut_reach_shelf.get(d1_content_hash, [])
                    d2_mut_reach: list[float, str, str] = mut_reach_shelf.get(d2_content_hash, [])
                    d1_mut_reach.append((mut_reach, d2_content_hash))
                    d2_mut_reach.append((mut_reach, d1_content_hash))
                    mut_reach_shelf[d1_content_hash] = d1_mut_reach
                    mut_reach_shelf[d2_content_hash] = d2_mut_reach

                    print(f"  Calculated mutual reachability for {output[2][0].ljust(32)} and {output[2][1].ljust(32)} (Progress: {num_outputs_received}/{similarities_total_count})", end="\r")
            
        end_time = datetime.now()
        print(f"\nMutual reachabilities calculated. (Time taken: {(end_time - start_time)})")

    def _unfurl_cluster_mut_reach(self, content_hash: str, root_deck_id: str) -> list[tuple[float, str, str]]:
        '''
        Obtains mutual reachability values on a per-deck basis (as opposed to a per deck cluster basis) from the mutual reachability shelf.
        '''
        with shelve.open(self.mut_reach_shelf_path, flag="w") as mut_reach_shelf:
            ch_mut_reaches: list[tuple[float, str, str]] = mut_reach_shelf[content_hash]
            del mut_reach_shelf[content_hash]

        output = []

        root_deck = self.original_decks[root_deck_id]
        d = self.decks_and_clusters_by_contents[content_hash]
        if isinstance(d, deck.DeckCluster):
            other_identical_decks = d.decks - {root_deck}
            for other_deck in other_identical_decks:
                # For two decks in the same cluster, MR = cluster's K distance
                # because distance always = 0 and each node's K is the same.
                output.append((d.k_distance, root_deck_id, other_deck.id))

        for mut_reach, other_ch in ch_mut_reaches:
            other_d = self.decks_and_clusters_by_contents[other_ch]
            if isinstance(other_d, deck.DeckCluster):
                other_decks = other_d.decks
            else:
                other_decks: set[deck.Deck] = {other_d}

            for other_deck in other_decks:
                output.append((mut_reach, root_deck_id, other_deck.id))

        return output
    
    def _calculate_mutual_reachabilities_for_deck(self, this_ch: str, others: Iterable[str], min_distances: dict[str, tuple[float, str]]):
        this_decklike = self.decks_and_clusters_by_contents[this_ch]
        this_id = this_decklike.id
        if isinstance(this_decklike, deck.DeckCluster):
            # For two decks in the same cluster, MR = cluster's K distance
            # because distance always = 0 and each node's K is the same.
            first_in_cluster = next(iter(this_decklike.decks))
            other_identical_decks = this_decklike.decks - {first_in_cluster}
            this_id = first_in_cluster.id
            for other_identical_deck in other_identical_decks:
                min_distances[other_identical_deck.id] = (min(this_decklike.k_distance, min_distances.get(other_identical_deck.id, (1,))[0]), this_id)

        with shelve.open(self.similarity_shelf_path, flag="r") as similarity_shelf:
            for other_ch in others:
                similarity = similarity_shelf[f"{min(this_ch, other_ch)},{max(this_ch, other_ch)}"]
                other_decklike = self.decks_and_clusters_by_contents[other_ch]
                mut_reach = max(this_decklike.k_distance, other_decklike.k_distance, 0.5 - similarity)

                if isinstance(other_decklike, deck.DeckCluster):
                    other_decks = other_decklike.decks
                else:
                    other_decks: set[deck.Deck] = {other_decklike}
                for other_deck in other_decks: 
                    min_distances[other_deck.id] = (min(mut_reach, min_distances.get(other_deck.id, (1,))[0]), this_id)

    def _build_spanning_tree(self):
        '''
        Uses Prim's algorithm to build a spanning tree out of the mutual reachability values.
        '''
        start_time = datetime.now()
        print("Building spanning tree...")

        # Set of content hashes that has been retrieved from the mutual reachability shelf
        calculated_content_hashes: set[str] = set() 
        # Set of decks that have been already added to the tree (to avoid loops)
        decks_in_tree: set[str] = set()
        # Set of content hashes not yet used in the tree
        content_hashes_to_calc = set(self.decks_and_clusters_by_contents.keys())
        min_distances = {}

        # Initialize tree root
        self.spanning_tree_root: deck.Deck = self.original_decks[next(iter(self.original_decks))]
        cluster_check = self.decks_and_clusters_by_contents[self.spanning_tree_root.contents_hash]
        if isinstance(cluster_check, deck.DeckCluster):
            self.spanning_tree_root = next(iter(cluster_check.decks))
        decks_in_tree.add(self.spanning_tree_root.id)
        content_hashes_to_calc.remove(self.spanning_tree_root.contents_hash)
        self._calculate_mutual_reachabilities_for_deck(self.spanning_tree_root.contents_hash, content_hashes_to_calc, min_distances)
        calculated_content_hashes.add(self.spanning_tree_root.contents_hash)

        while len(decks_in_tree) < len(self.original_decks):
            other_deck_id = min(min_distances, key=min_distances.get)
            mut_reach_dist, this_deck_id = min_distances[other_deck_id]
            if other_deck_id not in decks_in_tree:
                decks_in_tree.add(other_deck_id)
                del min_distances[other_deck_id]
                heapq.heappush(self.spanning_tree, (mut_reach_dist, self.original_decks[this_deck_id], self.original_decks[other_deck_id]))

                other_deck_contents_hash = self.original_decks[other_deck_id].contents_hash
                if other_deck_contents_hash not in calculated_content_hashes:
                    content_hashes_to_calc.remove(other_deck_contents_hash)
                    self._calculate_mutual_reachabilities_for_deck(other_deck_contents_hash, content_hashes_to_calc, min_distances)
                    calculated_content_hashes.add(other_deck_contents_hash)
                print(f"  Connected {other_deck_id} to the spanning tree (Progress: {len(decks_in_tree)}/{len(self.original_decks)})", end="\r")

        end_time = datetime.now()
        print(f"\nSpanning tree built. (Time taken: {(end_time - start_time)})")

        filename = self.SAVE_PATH + ".aaatree"
        print(f"Saving spanning tree to {filename}...")
        with open(filename, "wb") as file:
            pickler = pickle.Pickler(file)
            pickler.dump((self.spanning_tree_root, self.spanning_tree))

    def _hdbscan_hierarchical_cluster(self):
        """
        Main HDBSCAN* cluster production logic.
        First builds a hierarchy based on the mutual reachability graph,
        then makes a condensed tree based on our minimum cluster (archetype) size,
        then lastly chooses clusters from the condensed tree.

        I am indebted to this resource here: https://nbviewer.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb
        for teaching me how to implement this algorithm.
        """
        start_time = datetime.now()
        print("Beginning clustering of decks with HDBSCAN* method...")

        self.cluster_hierarchy = ClusterHierarchy(self.original_decks.values(), self.spanning_tree)
        self.cluster_hierarchy.build_hierarchy()

        self.cluster_hierarchy.condense_tree()

        self.cluster_hierarchy.select_clusters_cohesion(self.card_counter)

        end_time = datetime.now()
        print(f"Archetypes determined. (Time taken: {(end_time - start_time)})")

        self.clusters = {c.id: c for c in self.cluster_hierarchy.selected_clusters}
        self.rogue_decks = self.cluster_hierarchy.rogue_decks

        filename = self.SAVE_PATH + ".aaaclusters"
        print(f"Saving archetypes to {filename}...")
        with open(filename, "wb") as file:
            pickler = pickle.Pickler(file)
            pickler.dump((self.clusters, self.rogue_decks))

    def rename_archetypes(self):
        super().rename_archetypes()

        filename = self.SAVE_PATH + ".aaaclusters"
        print(f"Saving archetypes to {filename}...")
        with open(filename, "wb") as file:
            pickler = pickle.Pickler(file)
            pickler.dump((self.clusters, self.rogue_decks))

    def print_cluster_tree(self):
        filename = f"reports/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_cluster_tree.txt"
        with open(filename, "w") as file:
            for node, leaves in self.cluster_hierarchy.condensed_tree.items():
                file.write(f"{node}: {leaves}\n")
        print(f"Saved cluster tree report to {filename}.")

    def cluster(self):
        start_time = datetime.now()

        self._auto_cluster_identical_decks()

        self._build_initial_similarity_matrix()

        self._build_spanning_tree()

        self._hdbscan_hierarchical_cluster()

        end_time = datetime.now()
        print(f"\nFinished! (Total time taken: {(end_time - start_time)})")
        
