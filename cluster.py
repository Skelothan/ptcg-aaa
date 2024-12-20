from config import CONFIG 
import copy
from datetime import datetime
import deck
import itertools
import math
import multiprocessing as mp
import multiprocessing.queues as mpq


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
        self.decks = decks
        self.decks_and_clusters: dict[str, deck.DeckLike] = copy.copy(decks)
        self.similarities: dict[tuple[str, str], float] = {}

        self.greatest_similarity: float = -1
        self.most_similar_pair: tuple[str, str] = tuple()

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
                print(f"  Calculated similarity for {output[0][0].ljust(32)} and {output[0][1].ljust(32)} (Progress: {num_outputs_received}/{similarities_total_count})", end="\r")
                self.similarities[output[0]] = output[1]

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