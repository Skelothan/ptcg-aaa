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


def download_tournament_results():
    """
    Downloads tournament details and standings files from LimitlessTCG from an initial list in the raw_data/ directory and saves them to the data/ directory.
    The initial list is configurable as CONFIG["TOURNAMENT_LIST"]. Only downloads data from tournaments with CONFIG["TOURNAMENT_MIN_PLAYERS"]
    or more players.

    Delays by 0.1 second in between each request pair to avoid flooding LimitlessTCG's API.
    """
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


def load_decks_from_files() -> dict[str, deck.Deck]:
    """
    Reads decklist data from the data/ directory and creates a `deck.Deck` object for each one.
    """
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


def load_decks() -> tuple[dict[str, deck.Deck], deck.CardCounter]:
    """
    Read decklist data from the data/ directory and load them all into a `deck.CardCounter`.
    """
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


def suggest_k_threshold(decks: dict[str: deck.Deck], card_counter: deck.CardCounter):
    cluster_engine = cluster.UPGMAClusterEngine(card_counter, decks)

    cluster_engine._auto_cluster_identical_decks()

    suggestion = sorted(cluster_engine.decks_and_clusters.values(), key=lambda d: d.num_decks, reverse=True)[0].num_decks + 1
    print(f"K-threshold must be at least {suggestion} (probably oughta be more than that though)")


def compute_archetypes(decks: dict[str: deck.Deck], card_counter: deck.CardCounter) -> cluster.ClusterEngine:
    # TODO: make cluster engine configurable
    cluster_engine = cluster.HDBSCANClusterEngine(card_counter, decks)

    cluster_engine.cluster()

    return cluster_engine    


def main():
    decks = None
    card_counter = None
    cluster_engine = None

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
                == Download data == 
                D1. Download tournament results
                  
                == Process data == 
                P2. Load decks
                  
                0. Exit
                """)

            option = input("> ")
            
            if option == "d1":
                download_tournament_results()
            elif option == "p2":
                decks, card_counter = load_decks()
            else:
                print("Goodbye!")
                exit(0)
        elif cluster_engine is None:
            print("""Please choose an option:
                == Download data == 
                D1. Download tournament results
                  
                == Process data == 
                P1. Suggest K-threshold
                P2. Load decks
                P3. Compute deck archetypes
                  
                == Reports ==
                R1. Print card usage report
                
                0. Exit
                """)

            option = input("> ")
            
            if option == "d1":
                download_tournament_results()
            elif option == "p1":
                suggest_k_threshold(decks, card_counter)
            elif option == "p2":
                decks, card_counter = load_decks()
            elif option == "p3":
                cluster_engine = compute_archetypes(decks, card_counter)
            elif option == "r1":
                print_card_usage_report(card_counter)
            else:
                print("Goodbye!")
                exit(0)
        else:
            print("""Please choose an option:
                == Download data == 
                D1. Download tournament results
                  
                == Process data == 
                P1. Suggest K-threshold
                P2. Load decks
                P3. Compute deck archetypes
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
            
            if option == "d1":
                download_tournament_results()
            elif option == "p1":
                suggest_k_threshold()
            elif option == "p2":
                decks, card_counter = load_decks()
            elif option == "p3":
                cluster_engine = compute_archetypes(decks, card_counter)
            elif option == "p4":
                cluster_engine.rename_archetypes()
            elif option == "r0":
                print_card_usage_report(card_counter)
                cluster_engine.print_cluster_report()
                cluster_engine.print_rogue_deck_report()
                cluster_engine.print_metagame_report()
            elif option == "r1":
                print_card_usage_report(card_counter)
            elif option == "r2":
                cluster_engine.print_cluster_report()
            elif option == "r3":
                cluster_engine.print_rogue_deck_report()
            elif option == "r4":
                cluster_engine.print_metagame_report()
            else:
                print("Goodbye!")
                exit(0)
    

if __name__ == "__main__":
    main()