from __future__ import annotations

import abc
import hashlib
import math
import re
from collections import Counter
from datetime import date
from functools import lru_cache

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

# Normalizes set codes
SET_CODE_MAP = {
    "SVP": "PR-SV",
    "SP": "PR-SW",
    # Correcting Limitless's unofficial set codes
    "E1": "EX",
    "E2": "AQ",
    "E3": "SK",
    # Following not official for promos but would be consistent with others
    "SMP": "PR-SM",
    "XYP": "PR-XY",
    "BWP": "PR-BW",
    "HSP": "PR-HS",
    "DPP": "PR-DP",
    "NP": "PR-N",
    "WP": "PR-W",
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
    "Max Rod",
    "Treasure Tracker",
}


class DeckLike(metaclass=abc.ABCMeta):
    """
    Base class for `Deck`s and `DeckCluster`s.
    """

    def __init__(self):
        self._k_distance: float | None = None
        raise NotImplementedError

    @property
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
    
    @property
    @abc.abstractmethod
    def k_distance(self) -> float | None:
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
    
def is_special_darkness_metal(card_name, set_code, number):
    """
    Checks if a card is Special Darkness or Special Metal Energy.

    Parameters
    ----------
    card_name : str
        The card's name.
    set_code : str
        The card's set code.
    number : int
        The card's collector number.

    Returns
    -------
    bool
        True if the provided card name, set, and number refer to Special Darkness Energy or Special Metal Energy; False otherwise
    """
    if card_name == "Darkness Energy" \
        and (set_code, number) in {
            ("N1", 104),
            ("EX", 158),
            ("AQ", 142),
            ("RS", 93),
            ("EM", 86),
            ("UF", 96),
            ("DS", 103),
            ("HP", 94),
            ("PK", 87),
            ("MT", 119),
            ("SW", 129),
            ("MD", 93),
            ("RR", 99),
            ("UD", 79),
            ("CL", 86),
        }:
        return True
    elif card_name == "Metal Energy" \
        and (set_code, number) in {
            ("N1", 19),
            ("EX", 159),
            ("AQ", 143),
            ("RS", 94),
            ("EM", 88),
            ("UF", 97),
            ("DS", 107),
            ("HP", 95),
            ("PK", 88),
            ("MT", 120),
            ("SW", 130),
            ("MD", 95),
            ("RR", 100),
            ("UD", 80),
            ("CL", 87),
        }:
        return True
    else:
        return False

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

        # Used during HDBSCAN* clustering
        self.death_distance: float
        self.k_most_similarities: list[tuple[float, str]] = []
        self._k_distance: float | None = None

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

        Will normalize Trainer and Energy card names during import.

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
            card_name = f"{card['name']} {card['set']} {card['number']}"
            if card_name in REPRINTS.keys():
                card_name = REPRINTS.get(card_name)
            decklist_dict[card_name] = card.get("count")

        for card in decklist.get("trainer"):
            decklist_dict[card["name"]] = card.get("count")

        for card in decklist.get("energy"):
            card_name = card["name"]
            if is_special_darkness_metal(card["name"], card["set"], int(card["number"])):
                card_name = "Special " + card_name
            elif card_name in BASIC_ENERGY_NAMES:
                card_name = "Basic " + card_name
            decklist_dict[card_name] = card.get("count")

        self._decklist = Counter(decklist_dict)

    @property
    def k_distance(self) -> float | None:
        return self._k_distance
        
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
        
    @property
    def contents_hash(self) -> str:
        return hashlib.blake2b(
            str(sorted(self.decklist.items())).encode("UTF-8"),
            digest_size=16,
        ).hexdigest()

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
        self._k_distance: float | None = None

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
    
    @property
    def contents_hash(self) -> str:
        return hashlib.blake2b(
            str(sorted(Counter({k: int(round(v,0)) if math.isclose(v, round(v,0)) else v for k, v in self.decklist.items()}).items())).encode("UTF-8"),
            digest_size=16,
        ).hexdigest()
    
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
    
    @property
    def k_distance(self) -> float | None:
        return self._k_distance
    

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
        # elif self.get_card_type(card) == "Basic Energy":
            # return 2 / DECK_SIZE # doing the more correct 59 doesn't weight down basics enough. Archetypes probably aren't determined by energy type.
            #return 59/60
        else:
            return 4 / DECK_SIZE

    @lru_cache(maxsize=128)
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

    def weight_cards_by_max_possible_usage_adjusted(self, cards: Counter[str, int]) -> Counter[str, float]:
        """
        Adjusts the value of each card in `cards` by reducing the value of more-included cards.

        Adjusted from the previous version in a few ways:
        * The count of Energy cards is scaled before being used. This is to reduce the chance that a deck becomes highly defined by large counts of Energy (e.g. Ceruledge ex decks)

        Parameters
        ----------
        cards : collections.Counter[str, int]
            A Counter collection of card names to counts, most likely a decklist.
        
        Returns
        -------
        collections.Counter[str, float]
            A Counter with the same keys, but with values reduced more for more popular cards.
        """
        cards_weighted: Counter[str, float] = {}
        for k, v in cards.items():
            if self.get_card_type(k) == "Basic Energy":
                cards_weighted[k] = (v**(2/3)) * (1 - self.get_card_percent_of_max_usage(k))
            else:
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
    
    def get_deck_max_possible_inclusion_adjusted_weighted_Jaccard(self, deck1: DeckLike, deck2: DeckLike) -> float:
        """
        Returns Jaccard index of two decks where both decks are adjusted for card inclusion first and then for Energy.

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
            self.weight_cards_by_max_possible_usage_adjusted(overlap).total()
            / self.weight_cards_by_max_possible_usage_adjusted(deck1.decklist + deck2.decklist).total()
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
