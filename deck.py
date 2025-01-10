from __future__ import annotations

import abc
import hashlib
import heapq
import re
from collections import Counter
from datetime import date

import yaml

from config import CONFIG

DECK_SIZE = 60

BASIC_ENERGY_NAMES = {
    "Grass Energy",
    "Fire Energy",
    "Water Energy",
    "Lightning Energy",
    "Fighting Energy",
    "Psychic Energy",
    "Darkness Energy",
    "Metal Energy",
}

SET_CODE_MAP = {
    "SVP": "PR-SV",
    "SP": "PR-SW"
}

ACE_SPECS = {
    "Computer Search",
    "Crystal Edge",
    "Crystal Wall",
    "Gold Potion",
    "Dowsing Machine",
    "Scramble Switch",
    "Victory Piece",
    "Life Dew",
    "Rock Guard",
    "G Booster",
    "G Scope",
    "Master Ball",
    "Scoop Up Cyclone",
    "Awakening Drum",
    "Hero's Cape",
    "Maximum Belt",
    "Prime Catcher",
    "Reboot Pod",
    "Neo Upper Energy",
    "Hyper Aroma",
    "Secret Box",
    "Survival Brace",
    "Unfair Stamp",
    "Legacy Energy",
    "Dangerous Laser",
    "Neutralization Zone",
    "Poké Vital A",
    "Deluxe Bomb",
    "Grand Tree",
    "Sparkling Crystal",
    "Amulet of Hope",
    "Brilliant Blender",
    "Energy Search Pro",
    "Megaton Blower",
    "Miracle Headset",
    "Precious Trolley",
    "Scramble Switch",
    "Enriching Energy",
}


class DeckLike(metaclass=abc.ABCMeta):
    """
    Base class for `Deck`s and `DeckCluster`s.
    """

    def __init__(self):
        self.similarities: list[tuple[float, Deck]]
        raise NotImplementedError

    @abc.abstractmethod
    def contents_hash() -> int:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def title(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def id(self) -> str:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def num_decks(self) -> int:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def decklist(self) -> Counter:
        raise NotImplementedError
    
    def __eq__(self, other: DeckLike):
        return isinstance(other, DeckLike) and self.id == other.id
    
    def __ne__(self, other: DeckLike):
        return not isinstance(other, DeckLike) or self.id != other.id

    def __lt__(self, other: DeckLike):
        return self.id < other.id
    
    def __le__(self, other: DeckLike):
        return self.id <= other.id
    
    def __gt__(self, other: DeckLike):
        return self.id > other.id
    
    def __ge__(self, other: DeckLike):
        return self.id >= other.id


class Deck(DeckLike):
    """
    Class representing a deck someone played at a tournament.

    Attributes
    ----------
    raw_decklist : str
        A PTCGL-importable string decklist. The input data for this list.
    decklist : collections.Counter
        A list of every card in the deck, with counts. Extraneous information such as card number for Trainers and Energy is removed, and "Basic" prepended to Basic Energy cards.
    player_name : str
        The player's name. If not provided, "Unknown Player" will be used.
    tournament_name : str
        The name of the tournament the deck was played at. If not provided, "Unknown Tournament" will be used.
    date : datetime.date
        The date the tournament took place on. If not provided, the UNIX epoch will be used. Highly recommended as online tournaments tend to reuse the same name week to week.
    format : str
        The format the deck was played in, formatted as the starting set symbol and end set symbol. Example: SSH-CRZ, for the Sword & Shield to Crown Zenith format.
    title : str
        Returns a string with the deck's player, tournament, and date.
    id : str
        Returns a 16-digit hexadecimal hash of the deck's `title` and `decklist`. Used to ensure a deck doesn't get counted twice.
    """

    def __init__(
        self,
        player_name="Unknown Player",
        tournament_name="Unknown Tournament",
        date=date(1970, 1, 1),
        format="Unknown Format",
    ):
        """
        Class constructor. 

        Parameters
        ----------
        player_name : str, optional
            The player's name. If not provided, "Unknown Player" will be used.
        tournament_name : str, optional
            The name of the tournament the deck was played at. If not provided, "Unknown Tournament" will be used.
        date : datetime.date, optional
            The date the tournament took place on. If not provided, the UNIX epoch will be used. Highly recommended as online tournaments tend to reuse the same name week to week.
        format : str, optional
            The format the deck was played in.
        """
        self.player_name = player_name
        self.tournament_name = tournament_name
        self.date = date
        self.format = format
        self.similarities = []
        self.mut_reach_similarities: list[tuple[float, Deck, Deck]] = []

        # Used during HDBSCAN* clustering
        self.death_distance: float

    def load_decklist_ptcgl(self, decklist: str):
        """
        Loads a PTCGL-compatible decklist into this Deck.

        Will normalize Trainer and Energy card names during import. Currently does not handle Special Darkness Energy or Special Metal Energy correctly, so don't use on retro decklists.

        Parameters
        ----------
        decklist : str
            A PTCGL-importable string decklist.
        """
        self.raw_decklist = decklist
        decklist_list = []

        card_types = decklist.split("\n\n")

        # Pokémon
        for card_count in card_types[0].split("\n")[1:]:
            card = card_count.split(" ", 1)
            decklist_list += [card[1]] * (int)(card[0])

        # Trainers
        for card_count in card_types[1].split("\n")[1:]:
            card = card_count.split(" ")
            card_name = " ".join(card[1:-2])
            decklist_list += [card_name] * (int)(card[0])

        # Energy
        for card_count in card_types[2].split("\n")[1:]:
            card = card_count.split(" ")
            card_name = " ".join(card[1:-2])
            if card_name in BASIC_ENERGY_NAMES:
                card_name = "Basic " + card_name
            decklist_list += [card_name] * (int)(card[0])

        self._decklist = Counter(decklist_list)

    def load_decklist_limitless(self, decklist: dict):
        """
        Loads a Limitless API-format decklist into this Deck.

        Will normalize Trainer and Energy card names during import. Currently does not handle Special Darkness Energy or Special Metal Energy correctly, so don't use on retro decklists.

        Parameters
        ----------
        decklist : str
            A decklist from Limitless TCG's API.
        """
        decklist_dict = {}

        with open("reprint_list.yml", "r") as reprint_file:
            REPRINTS = yaml.safe_load(reprint_file)

        for card in decklist.get("pokemon"):
            if card["set"] in SET_CODE_MAP.keys():
                card["set"] = SET_CODE_MAP.get(card["set"])
            card_name = f"{card["name"]} {card["set"]} {card["number"]}"
            if card_name in REPRINTS.keys():
                card_name = REPRINTS.get(card_name)
            decklist_dict[card_name] = card.get("count")

        for card in decklist.get("trainer"):
            decklist_dict[card["name"]] = card.get("count")

        for card in decklist.get("energy"):
            card_name = card["name"]
            if card_name in BASIC_ENERGY_NAMES:
                card_name = "Basic " + card_name
            decklist_dict[card_name] = card.get("count")

        self._decklist = Counter(decklist_dict)

    # Could try caching this somehow
    @property
    def k_distance(self) -> float:
        if len(self.similarities) < CONFIG["K_THRESHOLD"]:
            raise ValueError
        return 0.5 - heapq.nlargest(CONFIG["K_THRESHOLD"], self.similarities)[-1][0]

    def __repr__(self) -> str:
        return f"<Deck: {self.title}>"
    
    def __hash__(self) -> int:
        return hash(f"{self.title}\n{str(sorted(self.decklist.items()))}".encode("UTF-8"))
    
    def __add__(self, other: DeckLike) -> DeckLike:
        if isinstance(other, DeckCluster):
            other.add_deck(self)
            return other
        elif isinstance(other, Deck):
            cluster = DeckCluster()
            cluster.add_deck(self)
            cluster.add_deck(other)
            return cluster
        else:
            raise TypeError(f"Cannot add item of type {type(other)} to Deck")
        
    def contents_hash(self) -> int:
        return hash(str(sorted(self.decklist.items())))

    @property
    def title(self) -> str:
        return f"{self.player_name} @ {self.tournament_name}, {self.date}"

    @property
    def id(self) -> str:
        return hashlib.blake2b(
            f"{self.title}\n{str(sorted(self.decklist.items()))}".encode("UTF-8"),
            digest_size=16,
        ).hexdigest()
    
    @property
    def num_decks(self) -> int:
        return 1
    
    @property
    def decklist(self) -> Counter:
        return self._decklist

    
class DeckCluster(DeckLike):

    cluster_number = 1
    
    def __init__(self):
        self.number = DeckCluster.cluster_number
        self._id = f"cluster-{self.number}"
        self._title = f"Cluster {self.number}"
        DeckCluster.cluster_number += 1
        self.decks: set[Deck] = set()
        self.similarities = []

    def add_deck(self, deck: Deck):
        self.decks.add(deck)

    def rename(self, new_name):
        """
        Rename this DeckCluster.

        Parameters
        ----------
        new_name : str
            The new name for this CardCounter.
        """
        self._title = new_name

    def __repr__(self) -> str:
        return f"<DeckCluster: {self.title} ({self.num_decks})>"

    def __add__(self, other: DeckLike) -> DeckLike:
        if isinstance(other, Deck):
            self.add_deck(other)
            return self
        elif isinstance(other, DeckCluster):
            if self.number < other.number:
                self.decks.update(other.decks)
                return self
            else:
                return other + self
        else:
            raise TypeError(f"Cannot add item of type {type(other)} to DeckCluster")
        
    def __hash__(self) -> int:
        return hash(self.id.encode("UTF-8"))

    def __len__(self) -> int:
        return len(self.decks)
    
    def contents_hash(self) -> int:
        return hash(str(sorted(self.decklist.items())))

    @property
    def title(self) -> str:
        return self._title

    @property
    def id(self) -> str:
        return self._id

    @property
    def decklist(self) -> Counter:
        card_sum = Counter()

        for deck in self.decks:
            card_sum += deck.decklist

        for k,v in card_sum.items():
            card_sum[k] = v/len(self)

        return card_sum
    
    @property
    def num_decks(self) -> int:
        return len(self.decks)
    

class CardCounter:
    """
    A class that aggregates statistics from multiple `Deck`s.

    Attributes
    ----------
    name : str
        A display name for this object.
    num_decks : int
        The number of `Deck`s this `CardCounter` has tallied.
    added_deck_ids : Set[str]
        The `id`s of all the `Deck`s this `CardCounter` has tallied.
    card_counts : collections.Counter[str, int]
        The number of times a card has been used in all of the `Deck`s this `CardCounter` has tallied. Multiple copies add that many to each card's count.
    card_inclusions : collections.Counter[str, int]
        The number of decks each card has been used in in all of the `Deck`s this `CardCounter` has tallied. Multiple copies still only add 1 to each card's count.
    """

    def __init__(self, name="Untitled Card Counter"):
        """
        Class constructor.

        Parameters
        ----------
        name : str, optional
            A display name for this object.
        """
        self.name = name
        self.num_decks = 0
        self.added_deck_ids = set()
        self.card_counts = Counter()
        self.card_inclusions = Counter()

    def __repr__(self):
        return f"<CardCounter: {self.name}>"

    def add_deck(self, deck: Deck):
        """
        Add a deck to this `CardCounter`'s tallies.

        If the deck has been previously added, it will be skipped.
        """
        # Don't allow adding the same deck twice
        if deck.id in self.added_deck_ids:
            return

        self.card_counts += deck.decklist

        for card in set(deck.decklist):
            self.card_inclusions[card] += 1

        self.num_decks += 1
        self.added_deck_ids.add(deck.id)

    def get_card_list(self) -> set[str]:
        """
        Returns a set containing the names of every card found in the decks in this `CardCounter`.

        Returns
        -------
        set[str]
            Set containing all unique card names found in included decks.
        """
        return set(self.card_counts.keys())

    def get_card_count(self, card: str) -> int:
        """
        Returns the number of times a `card` has been seen among all `Deck`s this `CardCounter` has tallied.

        Counts duplicates separately. If there is one deck with four copies of Professor's Research, then this function would return 4.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        int
            The number of times that card has been seen.
        """
        return self.card_counts.get(card, 0)

    def get_card_inclusion(self, card: str) -> int:
        """
        Returns the number of times a `card` has been included among all `Deck`s this `CardCounter` has tallied.

        Counts duplicates once. If there is one deck with four copies of Professor's Research, then this function would return 1.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        int
            The number of times that card has been included.
        """
        return self.card_inclusions[card]

    def get_card_usage_ratio(self, card: str) -> float:
        """
        Returns the use of a card compared to all other cards, as a decimal.

        Counts duplicates separately. If there is one deck with four copies of Professor's Research, then this function would return 4/60 = 0.0666666666667.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        float
            The ratio of this card's usage compared to all other cards in the surveyed decks.
        """
        return self.get_card_count(card) / self.card_counts.total()
    
    def max_possible_usage(self, card: str) -> float:
        """
        Returns the maximum usage ratio for the given card.

        If four copies are allowed, and the deck size is 60, then returns 4/60. If only one copy is allowed, returns 1/60.
        Treats the max number of copies of Basic Energy as 4; using the actual theoretical max would overweight them.

        Does not correctly restrict Gen 2 Shining Pokémon. Don't use on retro decklists!

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        float
            The maximum proportion of a deck this card could take up.
        """
        if any((
            "Radiant" in card,
            "Prism Star" in card,
            self.get_card_type(card) == "Pokémon" and " Star" in card,
            card in ACE_SPECS,
        )):
            return 1 / DECK_SIZE
        elif self.get_card_type(card) == "Basic Energy":
            return 2 / DECK_SIZE # doing the more correct 59 doesn't weight down basics enough. Archetypes probably aren't determined by energy type.
            #return 59/60
        else:
            return 4 / DECK_SIZE

    # Could put a LRU cache on this function
    def get_card_percent_of_max_usage(self, card: str) -> float:
        """
        Returns the ratio of a card's actual usage to its maximum possible usage.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        float
            The ratio of a card's actual usage to its maximum possible usage. Range [0, 1].
        """
        return self.get_card_usage_ratio(card) / self.max_possible_usage(card)
    
    def get_card_average_count(self, card: str) -> float:
        """
        Returns the average number of times a card is included in a deck.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        float
            The expected value of how many cards are included in any deck.
        """
        return self.get_card_count(card) / self.num_decks

    def get_card_inclusion_ratio(self, card: str):
        """
        Returns the inclusion of a card compared to all other cards, as a decimal.

        Put another way, what proportion of decks include at least one copy of this card?
        Counts duplicates once. If there is one deck with four copies of Professor's Research, then this function would return 1/1 = 1.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        float
            The ratio of decks including this card to all decks.
        """
        return self.get_card_inclusion(card) / self.num_decks
    
    def diff_cards_by_average_count(self, cards: Counter[str, int]) -> Counter[str, float]:
        """
        Adjusts the value of each card in `cards` to be the difference between its original value and the average number of that card seen in all decks

        Parameters
        ----------
        cards : collections.Counter[str, int]
            A Counter collection of card names to counts, most likely a decklist.
        
        Returns
        -------
        collections.Counter[str, float]
            A Counter with the same keys, but with values reduced more for more popular cards.
        """
        cards_weighted = {}
        for k, v in cards.items():
            cards_weighted[k] = v - self.get_card_average_count(k)
        return Counter(cards_weighted)

    def weight_cards_by_inclusion(self, cards: Counter[str, int]) -> Counter[str, float]:
        """
        Adjusts the value of each card in `cards` by reducing the value of more-included cards.

        If a card is included in every deck, it'll get weighted down to 0. If it's included in only one deck, it'll retain nearly all of its original value.

        Parameters
        ----------
        cards : collections.Counter[str, int]
            A Counter collection of card names to counts, most likely a decklist.
        
        Returns
        -------
        collections.Counter[str, float]
            A Counter with the same keys, but with values reduced more for more popular cards.
        """
        cards_weighted = {}
        for k, v in cards.items():
            cards_weighted[k] = v * (1 - self.get_card_inclusion_ratio(k))
        return Counter(cards_weighted)
    
    def weight_cards_by_max_possible_usage(self, cards: Counter[str, int]) -> Counter[str, float]:
        """
        Adjusts the value of each card in `cards` by reducing the value of more-included cards.

        If a card is included in every deck, it'll get weighted down to 0. If it's included in only one deck, it'll retain nearly all of its original value.

        Parameters
        ----------
        cards : collections.Counter[str, int]
            A Counter collection of card names to counts, most likely a decklist.
        
        Returns
        -------
        collections.Counter[str, float]
            A Counter with the same keys, but with values reduced more for more popular cards.
        """
        cards_weighted = {}
        for k, v in cards.items():
            cards_weighted[k] = v * (1 - self.get_card_percent_of_max_usage(k))
        return Counter(cards_weighted)

    def get_deck_Jaccard(_self, deck1: DeckLike, deck2: DeckLike) -> float:
        """
        Returns the Jaccard index of two decks.

        The Jaccard index measures the similarity of two sets, or in this case multisets. This value is 0.5 when both decks are identical and 0 when they have no cards in common.

        Parameters
        ----------
        deck1, deck2 : Deck
            The two decks to compare.
                
        Returns
        -------
        float
            The Jaccard index of the decks. Ranges from [0.0, 0.5]
        """
        overlap = deck1.decklist & deck2.decklist
        return overlap.total() / (DECK_SIZE * 2)

    def get_deck_inclusion_weighted_Jaccard(self, deck1: DeckLike, deck2: DeckLike) -> float:
        """
        Returns Jaccard index of two decks where both decks are adjusted for card inclusion first.

        Cards used in a larger number of decks will be considered less when comparing the similarity of the two decks.
        This value is 0.5 when both decks are identical and 0 when they have no cards in common.

        Parameters
        ----------
        deck1, deck2 : Deck
            The two decks to compare.
                
        Returns
        -------
        float
            The inclusion-weighted Jaccard index of the decks. Ranges from [0.0, 0.5]
        """
        overlap = deck1.decklist & deck2.decklist
        return (
            self.weight_cards_by_inclusion(overlap).total()
            / self.weight_cards_by_inclusion(deck1.decklist + deck2.decklist).total()
        )
    
    def get_deck_max_possible_inclusion_weighted_Jaccard(self, deck1: DeckLike, deck2: DeckLike) -> float:
        """
        Returns Jaccard index of two decks where both decks are adjusted for card inclusion first.

        Cards used in a larger number of decks will be considered less when comparing the similarity of the two decks.
        This value is 0.5 when both decks are identical and 0 when they have no cards in common.

        Parameters
        ----------
        deck1, deck2 : Deck
            The two decks to compare.
                
        Returns
        -------
        float
            The inclusion-weighted Jaccard index of the decks. Ranges from [0.0, 0.5]
        """
        overlap = deck1.decklist & deck2.decklist
        return (
            self.weight_cards_by_max_possible_usage(overlap).total()
            / self.weight_cards_by_max_possible_usage(deck1.decklist + deck2.decklist).total()
        )

    def get_card_type(_self, card: str) -> str:
        """
        Returns a card's type based on its name. This is to avoid calling the Limitless API to check.

        Can differentiate between Basic and Special Energy. Not currently able to determine a Pokémon's evolution stage or other properties (e.g. V, ex) or Trainer card subtypes.

        Parameters
        ----------
        card : str
            The string name of the card. Pokémon cards should have their set symbol and number included. Trainer and Energy cards should have them removed.
        
        Returns
        -------
        str
            Whether this card is a Pokémon, Trainer, Basic Energy, or Special Energy.
        """
        if card.split(" ")[-1] == "Energy":
            if card.replace("Basic ", "", 1) in BASIC_ENERGY_NAMES:
                return "Basic Energy"
            else:
                return "Special Energy"
        elif re.match(r".*[a-zA-Z\-]{2,5} \d+$", card):
            return "Pokémon"
        return "Trainer"
    

# counter = CardCounter()

# linney_darkzard_deck = Deck(
#     decklists.linney_darkzard,
#     "Phillip Linney",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# yuiti_darkzard_deck = Deck(
#     decklists.yuiti_darkzard,
#     "Rafael Yuiti",
#     "Regional Joinville 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# gonzalez_darkzard_deck = Deck(
#     decklists.gonzalez_darkzard,
#     "Manuel Gonzalez",
#     "Regional Joinville 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# wilton_darkzard_deck = Deck(
#     decklists.wilton_darkzard,
#     "Pearce Wilton",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# reddy_lugia_deck = Deck(
#     decklists.reddy_lugia,
#     "Rahul Reddy",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# okada_dragapult_deck = Deck(
#     decklists.okada_dragapult,
#     "Ryuki Okada",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# van_kampen_gholdengo_deck = Deck(
#     decklists.van_kampen_gholdengo,
#     "Jelle van Kampen",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# ivanoff_palkia_deck = Deck(
#     decklists.ivanoff_palkia,
#     "Stéphane Ivanoff",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# hausmann_ogerbolt_deck = Deck(
#     decklists.hausmann_ogerbolt,
#     "Jan Hausmann",
#     "Regional Dortmund 2024",
#     date=date(2024, 9, 28),
#     format="BRS-SCR",
# )
# tord_pidgeot_control_deck = Deck(
#     decklists.tord_pidgeot_control,
#     "Tord Reklev",
#     "Regional Baltimore 2024",
#     date=date(2024, 9, 14),
#     format="BRS-SFA",
# )
# rojas_duk_gardevoir_deck = Deck(
#     decklists.rojas_duk_gardevoir,
#     "Miguel Angel Rojas Duk",
#     "Special Event Lima 2024",
#     date=date(2024, 10, 5),
#     format="BRS-SCR",
# )
# connor_quad_thorns_deck = Deck(
#     decklists.connor_quad_thorns,
#     "Calvin Connor",
#     "Late Night 210",
#     date=date(2024, 10, 2),
#     format="BRS-SCR",
# )
# hu_bst_par_darkzard = Deck(
#     decklists.hu_bst_par_darkzard,
#     "Derek Hu",
#     "Regional Vancouver 2024",
#     date=date(2024, 3, 23),
#     format="BST-PAR",
# )

# counter.add_deck(linney_darkzard_deck)
# counter.add_deck(yuiti_darkzard_deck)
# counter.add_deck(gonzalez_darkzard_deck)
# counter.add_deck(wilton_darkzard_deck)
# counter.add_deck(reddy_lugia_deck)
# counter.add_deck(okada_dragapult_deck)
# counter.add_deck(van_kampen_gholdengo_deck)
# counter.add_deck(ivanoff_palkia_deck)
# counter.add_deck(hausmann_ogerbolt_deck)
# counter.add_deck(tord_pidgeot_control_deck)
# counter.add_deck(rojas_duk_gardevoir_deck)
# counter.add_deck(connor_quad_thorns_deck)


# # print(counter.get_card_use_ratio("Charizard ex OBF 125"))
# # print(counter.get_card_use_ratio("Mist Energy"))

# # print("Shared:", linney_darkzard_deck.decklist & yuiti_darkzard_deck.decklist)
# # print("Linney only:", linney_darkzard_deck.decklist - yuiti_darkzard_deck.decklist)
# # print("Yuiti only:", yuiti_darkzard_deck.decklist - linney_darkzard_deck.decklist)

# # print(counter.get_deck_similarity(linney_darkzard_deck, linney_darkzard_deck))
# # print(counter.get_deck_similarity(linney_darkzard_deck, yuiti_darkzard_deck))
# # print(counter.get_deck_similarity(linney_darkzard_deck, wilton_darkzard_deck))
# # print(counter.get_deck_similarity(linney_darkzard_deck, reddy_lugia_deck))
# # print(counter.get_deck_similarity(linney_darkzard_deck, okada_dragapult_deck))
# # print(counter.get_deck_similarity(reddy_lugia_deck, okada_dragapult_deck))
# # print(counter.get_deck_similarity(ivanoff_palkia_deck, hausmann_ogerbolt_deck))
# # print(counter.get_deck_similarity(ivanoff_palkia_deck, ivanoff_palkia_deck))
# # print(counter.get_deck_similarity(hausmann_ogerbolt_deck, hausmann_ogerbolt_deck))

# print(
#     counter.get_deck_inclusion_weighted_Jaccard(linney_darkzard_deck, hu_bst_par_darkzard)
# )
# print(
#     counter.get_deck_inclusion_weighted_Jaccard(linney_darkzard_deck, hu_bst_par_darkzard)
# )

# # print(counter.weight_cards_by_inclusion(tord_pidgeot_control_deck.decklist))
# # print(counter.weight_cards_by_inclusion(rojas_duk_gardevoir_deck.decklist))
