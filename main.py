from datetime import datetime, timezone
import copy
import csv
import deck
import itertools
import json
import logging
import math
import multiprocessing
import os.path
import requests
import threading
import time
from typing import Set, Union
import yaml

logger = logging.getLogger(__name__)


FORMAT_DATES = {
    # all formats BST-PAF
    "standard_e-g": (datetime(2023, 3, 30, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 3, 21, 17, 0, 0, 0, tzinfo=timezone.utc)), 
    # "BST-SVI":
    # "BST-PAL":
    # "BST-OBF":
    # "BST-MEW":
    # "BST-PAR":
    # "BST-PAF":

    # all formats BRS-PRE
    "standard_f-h": (datetime(2024, 3, 21, 17, 0, 0, 0, tzinfo=timezone.utc), datetime.now(tz=timezone.utc)),
    "BRS-TEF": (datetime(2024, 3, 21, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 5, 23, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-TWM": (datetime(2024, 5, 23, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 8, 1, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SFA": (datetime(2024, 8, 1, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 9, 12, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SCR": (datetime(2024, 9, 12, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 11, 7, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SSP": (datetime(2024, 11, 7, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2025, 1, 16, 17, 0, 0, 0, tzinfo=timezone.utc)),
    # "BRS-PRE": (datetime(2025, 1, 16, 17, 0, 0, 0, tzinfo=timezone.utc), datetime.now(tz=timezone.utc)),

    # "standard_g-i":
}

class ClusterEngine:

    def __init__(self, card_counter: deck.CardCounter, decks: dict[str: deck.Deck]):
        self.card_counter = card_counter
        self.decks = decks
        self.decks_and_clusters = copy.copy(decks)
        self.similarities = {}

        self.greatest_similarity = -1
        self.most_similar_pair = tuple()

    def _update_most_similar_pair(self):
        self.most_similar_pair = max(self.similarities, key=lambda x: self.similarities[x])
        self.greatest_similarity = self.similarities[self.most_similar_pair]

    def _auto_cluster_identical_decks(self):
        print("Auto-clustering identical decks...")
        original_size = len(self.decks_and_clusters)
        deck_contents_cache = {}
        for key, deck in self.decks_and_clusters.items():
            if deck.contents_hash() not in deck_contents_cache:
                deck_contents_cache[deck.contents_hash()] = deck
            else: # Combine the two
                deck_contents_cache[deck.contents_hash()] = deck_contents_cache[deck.contents_hash()] + deck

        self.decks_and_clusters = {d.id: d for d in deck_contents_cache.values()}
        print(f"Identical decks clustered. (Reduced from {original_size} to {len(self.decks_and_clusters)} decks)")

    def _fill_queue(self, tasks: multiprocessing.Queue, num_threads):
        similarities_to_calc = itertools.combinations(self.decks_and_clusters.values(), 2)
        num_tasks_queued = 0
        for pair in similarities_to_calc:
            tasks.put(pair)
            num_tasks_queued += 1
        stop_signals_queued = 0
        for signal in [None] * num_threads:
            tasks.put(signal)
            stop_signals_queued += 1

    def _compute_similarities(self, tasks: multiprocessing.Queue, output: multiprocessing.Queue):
        while True:
            pair = tasks.get(block=True)
            if pair is None:
                output.put(None)
                break
            d1, d2 = pair
            similarity = self.card_counter.get_deck_max_possible_inclusion_weighted_Jaccard(d1, d2) # TODO: make function choice configurable
            output.put(((d1.id, d2.id), similarity))

    def _build_initial_similarity_matrix(self):
        start_time = datetime.now()
        print("Building initial similarity matrix...")

        similarities_total_count = math.comb(len(self.decks_and_clusters), 2)
        

        manager = multiprocessing.Manager()
        tasks = manager.Queue()
        outputs = manager.Queue()

        processes = []

        # Start task queue-filling producer process
        producer_process = multiprocessing.Process(target=self._fill_queue, args=(tasks, CONFIG.get("NUM_THREADS")))
        processes.append(producer_process)
        producer_process.start()

        # Start task queue-emptying worker processes
        for _ in range(CONFIG.get("NUM_THREADS")):
            process = multiprocessing.Process(target=self._compute_similarities, args=(tasks, outputs))
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
        start_time = datetime.now()
        print("Beginning clustering of decks with UPGMA method...")

        manager = multiprocessing.Manager()
        tasks = manager.Queue()
        outputs = manager.Queue()

        merge_count = 0

        while self.greatest_similarity > CONFIG.get("CLUSTER_SIMILARITY_THRESHOLD"):
            # Merge the two most similar decks/clusters
            d1 = self.decks_and_clusters[self.most_similar_pair[0]]
            d2 = self.decks_and_clusters[self.most_similar_pair[1]]
            cluster = d1 + d2

            merge_count += 1
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
                    process = multiprocessing.Process(target=self._compute_similarities, args=(tasks, outputs))
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

            self._update_most_similar_pair()

        end_time = datetime.now()
        print(f"\nFinished clustering. (Time taken: {(end_time - start_time)})")

    def cluster_upgma(self):
        # Auto-cluster any identical decks
        self._auto_cluster_identical_decks()

        # Initial similarity matrix build
        self._build_initial_similarity_matrix()

        # UPGMA
        self._do_upgma()

    def print_cluster_report(self):
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


def download_tournament_results():
    with open(f"raw_data/{CONFIG.get("TOURNAMENT_LIST")}") as tournament_file:
        tournaments = json.load(tournament_file)
        tournaments_filtered = filter(
            lambda t: t.get("players") >= CONFIG.get("TOURNAMENT_MIN_PLAYERS") 
                and datetime.fromisoformat(t.get("date")) >= FORMAT_DATES.get(CONFIG.get("TOURNAMENT_FORMAT_FILTER"))[0] 
                and datetime.fromisoformat(t.get("date")) < FORMAT_DATES.get(CONFIG.get("TOURNAMENT_FORMAT_FILTER"))[1], 
            tournaments
        )
    
    for tournament in tournaments_filtered:

        # If we haven't downloaded the tournament details, download them
        details_path = f"data/{CONFIG.get("TOURNAMENT_FORMAT_FILTER")}/{tournament.get("id")}_details.json"
        if not os.path.isfile(details_path):
            logger.info(f"Downloading details for {tournament.get("name")} [{tournament.get("id")}]")
            details_response = requests.get(f"https://play.limitlesstcg.com/api/tournaments/{tournament.get("id")}/details?key={API_KEY}")
            with open(details_path, "w") as details_file:
                details_file.write(details_response.text)
            # Wait a bit, as to not flood Limitless's API
            time.sleep(0.1)

        # If we haven't downloaded the tournament deck lists, download them
        standings_path = f"data/{CONFIG.get("TOURNAMENT_FORMAT_FILTER")}/{tournament.get("id")}_standings.json"
        if not os.path.isfile(standings_path):
            with open(details_path, "r") as details_file:
                details = json.load(details_file)

            if details.get("decklists"):
                logger.info(f"Downloading standings for {tournament.get("name")} [{tournament.get("id")}]")
                standings_response = requests.get(f"https://play.limitlesstcg.com/api/tournaments/{tournament.get("id")}/standings?key={API_KEY}")
                with open(standings_path, "w") as standings_file:
                    standings_file.write(standings_response.text)
                # Wait a bit, as to not flood Limitless's API
                time.sleep(0.1)


def load_decks_from_files() -> set:
    decks = {}

    start_time = datetime.now()
    print("Loading decks from standings files...")
    dir_path = f"data/{CONFIG.get("TOURNAMENT_FORMAT_FILTER")}"
    for file_path in os.listdir(dir_path):
        filename = os.fsdecode(file_path)
        if filename.endswith("standings.json"):
            with open(f"{dir_path}/{file_path.replace("standings", "details")}", "r") as details_file:
                details = json.load(details_file)
            with open(f"{dir_path}/{file_path}", "r") as standings_file:
                standings = json.load(standings_file)
                load_message = f"\r  Loading {len(standings)} deck(s) from tournament {details.get("name")}"
                print(load_message + " " * (os.get_terminal_size().columns - len(load_message) - 24), end="")
                for player in standings:

                    d = deck.Deck(player_name=player.get("name"), tournament_name=details.get("name"), date=datetime.fromisoformat(details.get("date")), format=CONFIG.get("TOURNAMENT_FORMAT_FILTER"))
                    d.load_decklist_limitless(player.get("decklist"))
                    decks[d.id] = d

    end_time = datetime.now()
    print(f"\nFinished loading {len(decks)} decks. (Time taken: {(end_time - start_time)})")

    return decks


def load_decks():
    decks = load_decks_from_files()
    card_counter = deck.CardCounter(name=f"{CONFIG.get("TOURNAMENT_FORMAT_FILTER")} CardCounter")
    for d in decks.values():
        card_counter.add_deck(d)

    

    return decks, card_counter


def print_card_usage_report(card_counter: deck.CardCounter):

    filename = f"reports/{CONFIG.get("TOURNAMENT_FORMAT_FILTER")}_card_counts.csv"
    with open(filename, "w") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["Name", "Number used", "Average copies per deck", f"% of max usage", "Number of decks with card", f"% of decks with card"])
        csvwriter.writerows([
            [
                card,
                card_counter.get_card_count(card), 
                card_counter.get_card_average_count(card),
                card_counter.get_card_percent_of_max_usage(card),
                card_counter.get_card_inclusion(card),
                card_counter.get_card_inclusion_ratio(card)
            ] for card in card_counter.get_card_list()
        ])
    print(f"Saved CSV report to {filename}.")


def compute_archetypes(decks: dict[str: deck.Deck], card_counter: deck.CardCounter):
    cluster_engine = ClusterEngine(card_counter, decks)

    cluster_engine.cluster_upgma()
    cluster_engine.print_cluster_report()


def main():
    global API_KEY
    global CONFIG

    with open("api.yml", "r") as api_file:
        API_KEY = yaml.safe_load(api_file).get("API_KEY")
    
    with open("config.yml", "r") as config_file:
        CONFIG = yaml.safe_load(config_file)

    decks = None
    card_counter = None

    print("==== Pokémon TCG Automatic Archetype Analyzer ====")
    while True:
        print("")
        print(f"Currently loaded decks: {"None" if card_counter is None else card_counter.name}")
        print(f"Tournament list: {CONFIG.get("TOURNAMENT_LIST")}")
        print(f"Current format: {CONFIG.get("TOURNAMENT_FORMAT_FILTER")}")
        print(f"Minimum no. players: {CONFIG.get("TOURNAMENT_MIN_PLAYERS")}")
        print("")

        if card_counter is None:
            print("""Please choose an option:
                1. Download tournament results
                2. Load decks
                0. Exit
                """)

            option = input("> ")
            
            if option == "1":
                download_tournament_results()
            elif option == "2":
                decks, card_counter = load_decks()
            else:
                print("Goodbye!")
                exit(0)
        else:
            print("""Please choose an option:
                1. Download tournament results
                2. Load decks
                3. Print card usage report
                4. Compute deck archetypes
                0. Exit
                """)

            option = input("> ")
            
            if option == "1":
                download_tournament_results()
            elif option == "2":
                decks, card_counter = load_decks()
            elif option == "3":
                print_card_usage_report(card_counter)
            elif option == "4":
                compute_archetypes(decks, card_counter)
            else:
                print("Goodbye!")
                exit(0)
    


if __name__ == "__main__":
    main()