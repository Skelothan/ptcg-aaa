import cluster
from config import API_KEY, CONFIG 
import csv
from datetime import datetime, timezone
import deck
import json
import logging
import os.path
import requests
import time

logger = logging.getLogger(__name__)


PROGRAM_VERSION = (0, 1, 0)

FORMAT_DATES = {
    # "standard_e-g": (datetime(2023, 3, 30, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 3, 21, 17, 0, 0, 0, tzinfo=timezone.utc)), 
    # "BST-SVI":
    # "BST-PAL":
    # "BST-OBF":
    # "BST-MEW":
    # "BST-PAR":
    # "BST-PAF":

    # "standard_f-h": (datetime(2024, 3, 21, 17, 0, 0, 0, tzinfo=timezone.utc), datetime.now(tz=timezone.utc)),
    "BRS-TEF": (datetime(2024,  3, 21, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024,  5, 23, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-TWM": (datetime(2024,  5, 23, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024,  8,  1, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SFA": (datetime(2024,  8,  1, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024,  9, 12, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SCR": (datetime(2024,  9, 12, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 11,  7, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-SSP": (datetime(2024, 11,  7, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2025,  1, 16, 17, 0, 0, 0, tzinfo=timezone.utc)),
    "BRS-PRE": (datetime(2025,  1, 16, 17, 0, 0, 0, tzinfo=timezone.utc), datetime(2025,  3, 27, 17, 0, 0, 0, tzinfo=timezone.utc)),

    # "standard_g-i":
    # "SVI-JTG": (datetime(2025,  3, 27, 17, 0, 0, 0, tzinfo=timezone.utc), datetime.now(tz=timezone.utc))
}


def download_tournament_results():
    """
    Downloads tournament details and standings files from LimitlessTCG from an initial list in the raw_data/ directory and saves them to the data/ directory.
    The initial list is configurable as CONFIG["TOURNAMENT_LIST"]. Only downloads data from tournaments with CONFIG["TOURNAMENT_MIN_PLAYERS"]
    or more players.

    Delays by 0.1 second in between each request pair to avoid flooding LimitlessTCG's API.
    """
    print("Preparing list of tournaments from raw data...")

    with open(f"raw_data/{CONFIG.get('TOURNAMENT_LIST')}") as tournament_file:
        tournaments = json.load(tournament_file)
        tournaments_filtered = filter(
            lambda t: t.get("players") >= CONFIG.get("TOURNAMENT_MIN_PLAYERS") 
                and datetime.fromisoformat(t.get("date")) >= FORMAT_DATES.get(CONFIG.get("TOURNAMENT_FORMAT_FILTER"))[0] 
                and datetime.fromisoformat(t.get("date")) < FORMAT_DATES.get(CONFIG.get("TOURNAMENT_FORMAT_FILTER"))[1], 
            tournaments
        )
    
    start_time = datetime.now()
    print("Downloading tournament details and decklists from Limitless...")
    for tournament in tournaments_filtered:

        # If we haven't downloaded the tournament details, download them
        details_path = f"data/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}/{tournament.get('id')}_details.json"
        if not os.path.isfile(details_path):
            download_message = f"  Downloading details for {tournament.get('name')} [{tournament.get('id')}]"
            print(download_message + " " * (os.get_terminal_size().columns - len(download_message) - 24), end="\r")
            details_response = requests.get(f"https://play.limitlesstcg.com/api/tournaments/{tournament.get('id')}/details?key={API_KEY}")
            with open(details_path, "w") as details_file:
                details_file.write(details_response.text)
            # Wait a bit, as to not flood Limitless's API
            time.sleep(0.1)

        # If we haven't downloaded the tournament deck lists, download them
        standings_path = f"data/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}/{tournament.get('id')}_standings.json"
        if not os.path.isfile(standings_path):
            with open(details_path, "r") as details_file:
                details = json.load(details_file)

            if details.get("decklists"):
                download_message = f"  Downloading standings for {tournament.get('name')} [{tournament.get('id')}]"
                print(download_message + " " * (os.get_terminal_size().columns - len(download_message) - 24), end="\r")
                standings_response = requests.get(f"https://play.limitlesstcg.com/api/tournaments/{tournament.get('id')}/standings?key={API_KEY}")
                with open(standings_path, "w") as standings_file:
                    standings_file.write(standings_response.text)
                # Wait a bit, as to not flood Limitless's API
                time.sleep(0.1)

    end_time = datetime.now()
    print(f"\nDone. (Time taken: {end_time - start_time})")


def load_decks_from_files() -> dict[str, deck.Deck]:
    """
    Reads decklist data from the data/ directory and creates a `deck.Deck` object for each one.
    """
    decks = {}

    start_time = datetime.now()
    tournament_count = 0
    print("Loading decks from standings files...")
    dir_path = f"data/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}"
    for file_path in os.listdir(dir_path):
        filename = os.fsdecode(file_path)
        if filename.endswith("standings.json"):
            with open(f"{dir_path}/{file_path.replace('standings', 'details')}", "r") as details_file:
                details = json.load(details_file)
            with open(f"{dir_path}/{file_path}", "r") as standings_file:
                standings = json.load(standings_file)
                tournament_count += 1
                load_message = f"\r  Loading {len(standings)} deck(s) from tournament {details.get('name')}"
                print(load_message + " " * (os.get_terminal_size().columns - len(load_message) - 24), end="")
                for player in standings:

                    d = deck.Deck(player_name=player.get("name"), tournament_name=details.get('name'), date=datetime.fromisoformat(details.get('date')), format=CONFIG.get('TOURNAMENT_FORMAT_FILTER'))
                    d.load_decklist_limitless(player.get("decklist"))
                    decks[d.id] = d

    end_time = datetime.now()
    print(f"\nFinished loading {len(decks)} decks from {tournament_count} tournaments. (Time taken: {(end_time - start_time)})")

    return decks


def load_decks() -> tuple[dict[str, deck.Deck], deck.CardCounter]:
    """
    Read decklist data from the data/ directory and load them all into a `deck.CardCounter`.
    """
    decks = load_decks_from_files()
    card_counter = deck.CardCounter(name=f"{CONFIG.get('TOURNAMENT_FORMAT_FILTER')} CardCounter")
    for d in decks.values():
        card_counter.add_deck(d)

    return decks, card_counter


def print_card_usage_report(card_counter: deck.CardCounter):

    filename = f"reports/{CONFIG.get('TOURNAMENT_FORMAT_FILTER')}_card_counts.csv"
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


def suggest_k_threshold(decks: dict[str: deck.Deck], card_counter: deck.CardCounter):
    cluster_engine = cluster.UPGMAClusterEngine(card_counter, decks)

    cluster_engine._auto_cluster_identical_decks()

    suggestion = sorted(cluster_engine.decks_and_clusters.values(), key=lambda d: d.num_decks, reverse=True)[0].num_decks + 1
    print(f"K-threshold must be at least {suggestion} (probably oughta be more than that though)")


def main():
    decks: dict[str, deck.Deck] = None
    card_counter: deck.CardCounter = None
    cluster_engine: cluster.HDBSCANClusterEngine = None

    print("==== Pokémon TCG Automatic Archetype Analyzer ====")
    while True:
        print("")
        print(f"Currently loaded decks: {'None' if card_counter is None else card_counter.name}")
        print(f"Tournament list: {CONFIG.get('TOURNAMENT_LIST')}")
        print(f"Current format: {CONFIG.get('TOURNAMENT_FORMAT_FILTER')}")
        print(f"Minimum no. players: {CONFIG.get('TOURNAMENT_MIN_PLAYERS')}")
        print("")

        if card_counter is None:
            print("""Please choose an option:
                == Download data == 
                D1. Download tournament results
                  
                == Process data == 
                P2. Load decks
                  
                X. Exit
                """)
        elif cluster_engine is None:
            print("""Please choose an option:
                == Download data == 
                D1. Download tournament results
                  
                == Process data == 
                P1. Suggest K-threshold
                P2. Load decks
                P3. Compute deck archetypes
                  P3a. Calculate similarities
                  P3b. Build spanning tree
                  P3c. Determine archetypes
                  
                == Reports ==
                R1. Print card usage report
                
                X. Exit
                """)
        else:
            print("""Please choose an option:
                == Download data == 
                D1. Download tournament results
                  
                == Load data ==
                L1. 
                L2.
                L3.
                  
                == Process data == 
                P1. Suggest K-threshold
                P2. Load decks
                P3. Compute deck archetypes
                  P3a. Calculate similarities
                  P3b. Build spanning tree
                  P3c. Determine archetypes
                P4. Rename archetypes
                  
                == Reports ==
                R0. Print all reports
                R1. Print card usage report
                R2. Print archetype report
                R3. Print rogue deck report
                R4. Print metagame report
               [R5. TBD: Print deck spiciness report]
                
                X. Exit
                """)

        option = input("> ").lower()
        match option:
            case "d1":
                download_tournament_results()
            case "p1":
                suggest_k_threshold(decks, card_counter)
            case "p2":
                decks, card_counter = load_decks()
                cluster_engine = cluster.HDBSCANClusterEngine(card_counter, decks)
            case "p3":
                cluster_engine.cluster()
            case "p3a":
                cluster_engine._build_initial_similarity_matrix()
            case "p3b":
                cluster_engine._calculate_mutual_reachabilities()
                cluster_engine._build_spanning_tree()
            case "p3c":
                cluster_engine._hdbscan_hierarchical_cluster()
            case "p4":
                cluster_engine.rename_archetypes()
            case "r0":
                print_card_usage_report(card_counter)
                cluster_engine.print_cluster_report()
                cluster_engine.print_rogue_deck_report()
                cluster_engine.print_metagame_report()
            case "r1":
                print_card_usage_report(card_counter)
            case "r2":
                cluster_engine.print_cluster_report()
            case "r3":
                cluster_engine.print_rogue_deck_report()
            case "r4":
                cluster_engine.print_metagame_report()
            case _:
                print("Goodbye!")
                exit(0)
    

if __name__ == "__main__":
    main()